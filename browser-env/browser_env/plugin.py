"""BrowserEnvPlugin — a real browser exposed as an A2E ``env`` capability.

This is the host side of the browser environment. It wires three things
together behind the SDK's ``EnvPlugin`` contract:

    backend (playwright | scripted)   the actual browser
    BrowserTask / score_episode        the goal + deterministic scoring
    ExperienceStore                    the (s,a,r,s',done) transition log

Lifecycle rules this plugin obeys (and which are easy to get wrong):

* ``env/reset`` starts an EPISODE. It navigates the browser to the task's
  start URL and clears per-episode agent state. It NEVER restarts or closes
  the browser — the browser is process-scoped, started once, closed on
  ``teardown()``.
* ``env/step`` applies ONE action and returns
  (observation, reward, done, info). A malformed action is NOT fatal: the
  episode stays alive and the agent sees the error in the observation, so it
  can correct course. (The SDK raises on stepping an already-done episode —
  that is the SDK's contract, not ours.)
* Task DATA would move through the SDK's ``env/data/*`` plane, which is
  orthogonal to episodes; this plugin does not need it (the browser fetches
  its own pages), so the base class no-ops remain in force.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from a2e.caps.base.protocol import A2EMessage, A2EError, A2EErrorCode
from a2e.caps.env.plugin import EnvPlugin
from a2e.caps.env.protocol import (
    EnvErrorCode,
    EnvExpListRequest,
    EnvExpListResponse,
    EnvObservation,
    EnvState,
    EnvStatePush,
)

from browser_env import actions as A
from browser_env.backend import ActionResult, make_backend
from browser_env.state import BrowserEpisode, EpisodeRecorder, ExperienceStore
from browser_env.tasking import BrowserTask, TaskError, score_episode
from browser_env.domdiff import (
    checksum as domdiff_checksum,
    diff_captures as domdiff_captures,
)

logger = logging.getLogger(__name__)

#: Actions that end the episode by themselves.
_TERMINAL_ACTIONS = A.TERMINAL


class BrowserEnvPlugin(EnvPlugin):
    """A2E environment plugin driving a real (or scripted) browser."""

    name = "env"

    # ── construction ───────────────────────────────────────────────────────
    def __init__(self, host_instance, config: Any):
        super().__init__(host_instance, config)
        cfg = config or {}

        # The SDK's EnvPlugin manages its own episode object; we keep our own
        # richer browser-episode record alongside it, joined only by episode_id.
        self._ep: Optional[BrowserEpisode] = None
        self._task: Optional[BrowserTask] = None
        self._last_snapshot = None
        self._clicked_text: List[str] = []
        self._filled: Dict[str, str] = {}
        self._actions: List[str] = []

        # ── explicit config: required inputs fail loudly, never default ────
        self._backend_kind = str(cfg.get("BACKEND") or "").strip()
        if not self._backend_kind:
            raise ValueError(
                "BrowserEnvPlugin requires BACKEND in its plugin metadata "
                "(one of: 'playwright', 'scripted')."
            )
        self._root = str(cfg.get("ROOT") or "").strip()
        if not self._root:
            raise ValueError(
                "BrowserEnvPlugin requires ROOT in its plugin metadata "
                "(where the experience store is written)."
            )

        # ── optional config with documented, explicit defaults ─────────────
        self._default_max_steps = int(cfg.get("DEFAULT_MAX_STEPS", 15))
        self._max_text_chars = int(cfg.get("MAX_TEXT_CHARS", A.DEFAULT_TEXT_CHARS))
        self._max_elements = int(cfg.get("MAX_ELEMENTS", A.DEFAULT_MAX_ELEMENTS))
        self._step_penalty = float(cfg.get("STEP_PENALTY", 0.0))
        self._progress_reward = float(cfg.get("PROGRESS_REWARD", 0.0))
        self._screenshot_max_bytes = int(
            cfg.get("SCREENSHOT_MAX_BYTES", A.DEFAULT_SCREENSHOT_MAX_BYTES))
        self._dom_max_bytes = int(cfg.get("DOM_MAX_BYTES", A.DEFAULT_DOM_MAX_BYTES))
        self._console_max_entries = int(cfg.get("CONSOLE_MAX_ENTRIES", 40))
        self._browser_timeout_ms = int(cfg.get("BROWSER_TIMEOUT_MS", 15000))

        # ── experience store (transition log) ──────────────────────────────
        self._experiences = ExperienceStore(self._root)
        # ── episode recorder (JSONL replay artifact; off until started) ────
        self._recorder = EpisodeRecorder(self._root)
        # ── previous DOM capture, for mode="diff" (cleared per episode) ────
        self._dom_prev: Optional[Dict[str, Any]] = None
        # ── console entries already surfaced to the caller (for deltas) ────
        self._console_seen = 0

        # ── backend (browser), lazily started on first reset ───────────────
        backend_kwargs: Dict[str, Any] = {}
        if self._backend_kind == "playwright":
            backend_kwargs = {
                "headless": bool(cfg.get("HEADLESS", True)),
                "browser": cfg.get("BROWSER", "chromium"),
                "timeout_ms": self._browser_timeout_ms,
            }
            if cfg.get("USER_AGENT"):
                backend_kwargs["user_agent"] = cfg["USER_AGENT"]
            viewport = cfg.get("VIEWPORT")
            if isinstance(viewport, dict) and viewport:
                backend_kwargs["viewport"] = viewport
        self._backend = make_backend(self._backend_kind, **backend_kwargs)
        logger.info("browser env plugin ready (backend=%s root=%s)",
                    self._backend_kind, self._root)

    # ── lifecycle hooks (EnvPlugin contract) ───────────────────────────────
    def on_reset(self, seed: Optional[int], options: Dict[str, Any]) -> EnvState:
        """Begin an episode: resolve the task, navigate, snapshot.

        NOTE: the SDK creates its own ``_Episode`` *after* this returns, so
        ``self._episode`` is not available here — we must not call
        ``_require_episode()``.
        """
        options = options or {}
        raw_task = options.get("task") or options.get("task_name") or None

        if raw_task is not None:
            # Tasks live OUTSIDE the environment: the caller supplies the full
            # spec (goal + criteria) and the host binds it to this episode.
            # The host keeps no catalogue, so a bare name cannot be resolved.
            if not isinstance(raw_task, dict):
                raise TaskError(
                    "env/reset options.task must be the FULL task spec object "
                    f"(got {type(raw_task).__name__}). Tasks live outside the "
                    "environment — the host keeps no catalogue to resolve a "
                    "name against; pass name/start_url/goal/success inline."
                )
            task = BrowserTask.from_dict(raw_task)
            start_url = options.get("start_url") or task.start_url
            goal = options.get("goal") or task.goal
            max_steps = int(options.get("max_steps", task.max_steps))
        else:
            start_url = options.get("start_url") or options.get("url") or ""
            if not start_url:
                raise TaskError(
                    "env/reset requires either options.task (the full task "
                    "spec object supplied by the caller) or options.start_url."
                    " Neither was supplied."
                )
            task = BrowserTask(
                name=options.get("name", "adhoc"),
                start_url=start_url,
                goal=options.get("goal", "") or f"Browse starting at {start_url}",
                max_steps=int(options.get("max_steps", self._default_max_steps)),
                success={},  # ad-hoc reset declares no criteria -> scored on finish
            )
            goal = task.goal
            max_steps = task.max_steps

        # A fresh episode: start (once) and navigate the browser, clear state.
        result = self._backend.reset_session(
            start_url, timeout_ms=self._browser_timeout_ms)
        snapshot = self._backend.snapshot(self._max_text_chars, self._max_elements)

        self._task = task
        self._clicked_text = []
        self._filled = {}
        self._actions = []
        self._last_snapshot = snapshot
        # New episode: a DOM diff must not compare against the previous
        # episode's page.
        self._dom_prev = None
        # Console is episode-scoped: the backend cleared it on reset_session.
        self._console_seen = 0
        # A previous episode's recording is already stopped (or abandoned);
        # never let one episode append into another's artifact.
        if self._recorder.status().get("recording"):
            self._recorder.stop()
        self._ep = BrowserEpisode(
            episode_id="",  # assigned by the SDK's _Episode; filled on step
            goal=goal,
            start_url=start_url,
            url=snapshot.url or start_url,
            title=snapshot.title,
            max_steps=max_steps,
            visited=[snapshot.url or start_url],
            last_error="" if result.ok else result.error,
        )

        state = self._state_dict(snapshot)
        state["reset_ok"] = result.ok
        state["reset_detail"] = result.detail or result.error
        return EnvState(**state)

    def on_step(self, episode_id: str, action) -> EnvObservation:
        ep = self._require_episode()
        browser_ep = self._ep
        if browser_ep is None:
            raise RuntimeError("No browser episode; call reset() first.")
        if not browser_ep.episode_id:
            browser_ep.episode_id = episode_id

        step_num = int(getattr(ep, "step_num", 0)) + 1
        browser_ep.step_num = step_num
        browser_ep.last_error = ""

        try:
            norm = A.normalize_action(action)
        except A.ActionError as exc:
            browser_ep.last_error = str(exc)
            self._actions.append("invalid")
            obs = self._observation(
                episode_id, step_num, reward=self._step_penalty, done=False,
                extra={"action_error": str(exc)},
            )
            self._record(browser_ep, {"action": "invalid"}, self._step_penalty,
                         obs, done=False)
            self._write_trace("invalid", {}, obs, done=False, ok=False,
                              error=str(exc))
            return obs

        name = norm["name"]
        args = norm["args"]
        browser_ep.last_action = name

        if name not in A.ACTION_NAMES:
            err = f"unknown action {name!r}; valid actions: {A.ACTION_NAMES}"
            browser_ep.last_error = err
            self._actions.append(name)
            obs = self._observation(
                episode_id, step_num, reward=self._step_penalty, done=False,
                extra={"action_error": err},
            )
            self._record(browser_ep, {"action": name}, self._step_penalty,
                         obs, done=False)
            self._write_trace(name, args, obs, done=False, ok=False, error=err)
            return obs

        # ── dispatch ───────────────────────────────────────────────────────
        if name in _TERMINAL_ACTIONS:
            return self._finish(episode_id, step_num, args)

        try:
            result = self._dispatch(name, args)
        except A.ActionError as exc:
            browser_ep.last_error = str(exc)
            self._actions.append(name)
            obs = self._observation(
                episode_id, step_num, reward=self._step_penalty, done=False,
                extra={"action_error": str(exc)},
            )
            self._record(browser_ep, {"action": name, "args": args},
                         self._step_penalty, obs, done=False)
            self._write_trace(name, args, obs, done=False, ok=False,
                              error=str(exc))
            return obs

        self._actions.append(name)
        if not result.ok:
            browser_ep.last_error = result.error

        snapshot = self._backend.snapshot(self._max_text_chars, self._max_elements)
        self._last_snapshot = snapshot
        browser_ep.url = snapshot.url or browser_ep.url
        browser_ep.title = snapshot.title
        if snapshot.url and snapshot.url not in browser_ep.visited:
            browser_ep.visited.append(snapshot.url)

        # ── step reward + termination ──────────────────────────────────────
        reward = self._progress_reward if result.ok else self._step_penalty
        done = False
        metadata: Dict[str, Any] = {"action": name, "action_ok": result.ok}
        if name == "screenshot":
            browser_ep.screenshot_count += 1
            metadata["render"] = result.value
        elif name == "extract":
            if result.ok and result.value:
                browser_ep.extracted.append(str(result.value)[:2000])
            metadata["extract"] = str(result.value)[:4000] if result.ok else None
        elif name == "evaluate":
            metadata["value"] = result.value
        elif name == A.DOM:
            # Full capture goes back to the caller — they asked for the page.
            # (The JSONL trace keeps only a summary, see _trace_entry.)
            metadata["dom"] = (dict(result.value) if result.ok
                               else {"error": result.error})
        elif name == A.RECORD:
            metadata["record"] = (dict(result.value) if result.ok
                                  else {"error": result.error})
        elif name == A.CONSOLE:
            # Explicit read: return the whole buffer (not a delta).
            metadata["console"] = (dict(result.value) if result.ok
                                   else {"error": result.error})

        # Console output produced by THIS step. Omitted entirely when empty, so
        # a quiet step costs nothing on the wire — but a JS error or a page
        # warning surfaces immediately after the action that caused it. An
        # explicit `console` read already filled metadata["console"], and it
        # advanced the cursor, so this delta is empty and never overwrites it.
        if name != A.CONSOLE:
            console_delta = self._console_delta()
            if console_delta.get("new"):
                metadata["console"] = console_delta

        if step_num >= browser_ep.max_steps:
            # Out of budget: terminate with whatever score the episode earned.
            return self._terminate(
                episode_id, step_num, reason="max_steps",
                metadata=metadata, step_reward=reward,
                action_name=name, action_args=args,
                action_ok=result.ok, action_error=result.error,
            )

        obs = self._observation(episode_id, step_num, reward=reward, done=done,
                                extra=metadata)
        self._record(browser_ep, {"action": name, "args": args}, reward, obs,
                     done=False)
        self._write_trace(name, args, obs, done=False, ok=result.ok,
                          error=result.error or "")
        return obs

    def on_close(self):
        """Episode cleanup. Deliberately does NOT touch the browser.

        The browser is process-scoped (see module docstring): closing an
        episode must leave the browser running so the next reset is cheap.
        """
        if self._ep is not None:
            logger.info("browser episode closed (episode_id=%s steps=%d)",
                        self._ep.episode_id, self._ep.step_num)
        # Flush the recording so the artifact is complete even if the caller
        # never issues a `record stop`.
        try:
            if self._recorder.status().get("recording"):
                self._recorder.stop()
        except Exception:
            logger.warning("recorder close failed", exc_info=True)

    def observe(self) -> EnvObservation:
        """Return the LIVE environment state.

        Overrides the SDK's version, which reads its own cached
        ``_episode.state`` — a snapshot that only ``step()`` refreshes. Because
        the browser is also driven through the tools surface (BrowserToolPlugin
        → ``_dispatch``), that cached copy goes stale the moment a tool acts.
        Reading the backend here keeps env/observe and the tool surface from
        ever disagreeing about where the browser is.
        """
        ep = self._require_episode()
        browser_ep = self._ep
        if browser_ep is None:
            return super().observe()
        try:
            snap = self._backend.snapshot(self._max_text_chars, self._max_elements)
            self._last_snapshot = snap
            browser_ep.url = snap.url or browser_ep.url
            browser_ep.title = snap.title
            if snap.url and snap.url not in browser_ep.visited:
                browser_ep.visited.append(snap.url)
        except Exception:  # pragma: no cover - never fail an observe
            logger.warning("observe snapshot refresh failed", exc_info=True)
        state = self._state_dict(self._last_snapshot)
        return EnvObservation(
            episode_id=ep.id,
            step_num=browser_ep.step_num,
            state=EnvState(**state),
            done=browser_ep.done,
            truncated=False,
            metadata={"render_text": (self._last_snapshot.render_text()
                                      if self._last_snapshot is not None else "")},
        )

    def teardown(self):
        """Process teardown: close the browser and the experience store."""
        try:
            if self._backend is not None:
                self._backend.close()
        except Exception:
            logger.exception("browser backend close failed")
        try:
            if self._experiences is not None:
                self._experiences.close()
        except Exception:
            logger.exception("experience store close failed")
        super().teardown()

    # ── action dispatch ────────────────────────────────────────────────────
    def _dispatch(self, name: str, args: Dict[str, Any]) -> ActionResult:
        b = self._backend
        if name == A.NAVIGATE:
            url = args.get("url")
            if not url:
                raise A.ActionError("navigate requires 'url'")
            return b.navigate(str(url), timeout_ms=self._browser_timeout_ms)

        if name in (A.CLICK, A.FILL):
            selector = A.resolve_target(args, name)
            if name == A.CLICK:
                res = b.click(selector, timeout_ms=self._browser_timeout_ms)
                if res.ok:
                    # remember the clicked label for clicked_text_contains
                    snap = self._last_snapshot
                    if snap is not None:
                        prefix = '[data-a2e-eid="'
                        if selector.startswith(prefix):
                            eid = selector[len(prefix):].rstrip('"]')
                            for e in snap.elements:
                                if str(e.index) == eid:
                                    self._clicked_text.append(
                                        e.text or e.value or e.name or "")
                                    break
                return res
            text = args.get("text")
            if text is None:
                raise A.ActionError("fill requires 'text'")
            res = b.fill(selector, str(text), timeout_ms=self._browser_timeout_ms)
            if res.ok:
                # remember by field name when the element declared one
                snap = self._last_snapshot
                field = args.get("name") or args.get("field") or ""
                if not field and snap is not None:
                    prefix = '[data-a2e-eid="'
                    if selector.startswith(prefix):
                        eid = selector[len(prefix):].rstrip('"]')
                        for e in snap.elements:
                            if str(e.index) == eid:
                                field = e.name or f"eid{e.index}"
                                break
                self._filled[str(field or "field")] = str(text)
            return res

        if name == A.PRESS:
            key = args.get("key")
            if not key:
                raise A.ActionError("press requires 'key'")
            selector = ""
            if args.get("selector") or args.get("index") is not None:
                selector = A.resolve_target(args, name)
            return b.press(str(key), selector)

        if name == A.SCROLL:
            direction = str(args.get("direction", "down")).lower()
            if direction not in ("down", "up", "top", "bottom"):
                raise A.ActionError(
                    f"scroll direction must be one of down/up/top/bottom, got {direction!r}")
            amount = int(args.get("amount", A.DEFAULT_SCROLL_AMOUNT))
            return b.scroll(direction, amount)

        if name == A.BACK:
            return b.back()
        if name == A.FORWARD:
            return b.forward()
        if name == A.RELOAD:
            return b.reload()

        if name == A.EXTRACT:
            sel = args.get("selector") or (
                A.resolve_target(args, name) if args.get("index") is not None else "")
            return b.extract(str(sel or "body"))

        if name == A.EVALUATE:
            js = args.get("js") or args.get("script")
            if not js:
                raise A.ActionError("evaluate requires 'js'")
            return b.evaluate(str(js))

        if name == A.WAIT_FOR:
            selector = A.resolve_target(args, name)
            timeout = int(args.get("timeout_ms", A.DEFAULT_WAIT_MS))
            return b.wait_for(selector, timeout_ms=timeout)

        if name == A.SCREENSHOT:
            full = bool(args.get("full_page", False))
            return self._backend.screenshot(full_page=full,
                                            max_bytes=self._screenshot_max_bytes)

        if name == A.DOM:
            return self._dom_capture(args)

        if name == A.RECORD:
            return self._record_control(args)

        if name == A.CONSOLE:
            return self._console_read(args)

        raise A.ActionError(f"unhandled action {name!r}")

    # ── console output ─────────────────────────────────────────────────────
    def _console_read(self, args: Dict[str, Any]) -> ActionResult:
        """Return the episode's console output (messages + uncaught errors).

        ``clear=True`` empties the buffer after reading, so a caller can read
        once and know it will not see those lines again. The default is a full
        read without clearing — reading is idempotent.
        """
        max_entries = int(args.get("max_entries", 200) or 200)
        res = self._backend.console(max_entries=max_entries,
                                    clear=bool(args.get("clear", False)))
        if res.ok:
            # An explicit read advances the cursor so the next step's delta
            # does not repeat these lines. With clear=True the buffer is empty
            # and counted from zero again — the cursor must follow, otherwise
            # `total - seen` goes negative and no future delta ever reports.
            self._console_seen = (0 if args.get("clear")
                                  else int(res.value.get("total", 0)))
        return res

    def _console_delta(self) -> Dict[str, Any]:
        """Console lines produced since the last read/step. Never raises.

        Returned entries are capped so a page logging thousands of lines cannot
        flood one observation; ``truncated`` says so rather than hiding it.
        """
        try:
            res = self._backend.console(max_entries=self._console_max_entries)
            if not res.ok:
                return {}
            total = int(res.value.get("total", 0))
            entries = list(res.value.get("entries") or [])
            delta_count = total - self._console_seen
            if delta_count <= 0:
                return {}
            new = entries[-delta_count:] if delta_count <= len(entries) else entries
            self._console_seen = total
            out = {"new": new[: self._console_max_entries],
                   "total": total}
            if len(new) > self._console_max_entries:
                out["truncated"] = True
                out["dropped"] = int(res.value.get("dropped", 0))
            if res.value.get("dropped"):
                out["dropped"] = res.value["dropped"]
            return out
        except Exception:  # console must never break a step
            logger.warning("console delta failed", exc_info=True)
            return {}

    # ── DOM capture / diff ─────────────────────────────────────────────────
    def _dom_capture(self, args: Dict[str, Any]) -> ActionResult:
        """Capture the style-flattened DOM (default) or diff it.

        mode = "capture" (default) → returns the capture with a ``checksum``.
        mode = "diff"              → captures again and reports the diff against
                                      the previous capture.

        The diff needs NO new wire type: it is an env action like any other, so
        a client can answer "did my click change the page?" by issuing
        ``dom`` before and after with ``{"mode": "diff"}`` on the second one.
        The host keeps the previous capture, so no client-side state is needed.
        """
        mode = str(args.get("mode", "capture")).lower()
        max_bytes = int(args.get("max_bytes", self._dom_max_bytes) or
                        self._dom_max_bytes)
        res = self._backend.html(max_bytes=max_bytes)
        if not res.ok:
            return res

        value = dict(res.value)
        value["checksum"] = domdiff_checksum(value)

        if mode == "diff":
            prev = self._dom_prev
            if not prev:
                return ActionResult(
                    False, "dom",
                    error="no previous capture to diff against; call dom "
                          "with mode=capture first")
            value["diff"] = domdiff_captures(prev, value)
            res.detail = f"{res.detail} | diff: {value['diff']['summary']}"
        elif mode != "capture":
            return ActionResult(False, "dom",
                                error=f"unknown dom mode {mode!r}; "
                                      f"expected capture|diff")

        # Remember this capture for the next diff (both modes advance it).
        self._dom_prev = value
        return ActionResult(True, "dom", detail=res.detail, value=value)

    # ── recording control ──────────────────────────────────────────────────
    def _record_control(self, args: Dict[str, Any]) -> ActionResult:
        """start / stop / status for the episode recording (JSONL artifact).

        ``mode`` (or ``action``) selects the operation; default is ``status``.
        The artifact is named after the episode, so it stays discoverable even
        if the agent never calls stop.
        """
        mode = str(args.get("mode") or args.get("action") or "status").lower()
        ep = self._ep
        if mode == "start":
            out = self._recorder.start(
                ep.episode_id if ep else "",
                include_screenshots=bool(args.get("include_screenshots", False)),
                root=self._root,
            )
            return ActionResult(True, "record", detail="started", value=out)
        if mode == "stop":
            out = self._recorder.stop()
            return ActionResult(True, "record", detail="stopped", value=out)
        if mode not in ("status", "status_only"):
            return ActionResult(False, "record",
                                error=f"unknown record mode {mode!r}; "
                                      f"expected start|stop|status")
        return ActionResult(True, "record", detail="status",
                            value=self._recorder.status())

    def _trace_entry(self, action_name: str, args: Dict[str, Any], obs,
                     *, done: bool, ok: bool, error: str = "") -> Optional[Dict[str, Any]]:
        """Build one JSONL record. Returns None when not recording."""
        if not self._recorder.status().get("recording"):
            return None
        ep = self._ep
        entry: Dict[str, Any] = {
            "step": obs.step_num,
            "action": action_name,
            "args": args,
            "url": ep.url if ep else "",
            "title": ep.title if ep else "",
            "ok": ok,
            "error": error,
            "reward": obs.reward,
            "done": done,
            "ts": time.time(),
        }
        # Console output since the last step (same delta the observation saw).
        console = (obs.metadata or {}).get("console")
        if console:
            entry["console"] = console
        # Opt-in frames: only when include_screenshots was requested at
        # start(), so a default recording costs no extra browser round-trips.
        if self._recorder.status().get("include_screenshots"):
            frame = self._backend.screenshot(
                full_page=False, max_bytes=self._screenshot_max_bytes)
            if frame.ok:
                entry["screenshot"] = frame.value
            else:
                entry["screenshot_error"] = frame.error
        return entry

    def _write_trace(self, action_name: str, args: Dict[str, Any], obs,
                     *, done: bool, ok: bool = True, error: str = "") -> None:
        """Append one step to the recording. No-op when recording is off."""
        try:
            entry = self._trace_entry(action_name, args, obs,
                                      done=done, ok=ok, error=error)
            if entry is not None:
                self._recorder.record(entry)
        except Exception:  # a recording failure must never break an episode
            logger.warning("recording failed", exc_info=True)

    # ── terminal paths ─────────────────────────────────────────────────────
    def _finish(self, episode_id: str, step_num: int,
                args: Dict[str, Any]) -> EnvObservation:
        return self._terminate(episode_id, step_num, reason="finish",
                               metadata={"action": A.FINISH,
                                         "answer": str(args.get("answer", ""))},
                               step_reward=0.0,
                               action_name=A.FINISH, action_args=args)

    def _terminate(self, episode_id: str, step_num: int, *,
                   reason: str, metadata: Dict[str, Any],
                   step_reward: float,
                   action_name: str = "", action_args: Optional[Dict[str, Any]] = None,
                   action_ok: bool = True,
                   action_error: str = "") -> EnvObservation:
        """Score the episode, mark it done, emit the terminal reward push."""
        browser_ep = self._ep
        assert browser_ep is not None
        # Mark done BEFORE scoring: score_episode's `finished` criterion reads
        # the record, and this episode is terminating. Scoring first would
        # report finished=False on the very step that finishes it.
        browser_ep.done = True
        record = self._episode_record()
        scored = score_episode(self._task, record)
        score = float(scored["score"])
        passed = bool(scored["passed"])

        browser_ep.success = passed

        obs = self._observation(episode_id, step_num, reward=score, done=True,
                                extra={**metadata,
                                       "terminal_reason": reason,
                                       "score": score,
                                       "passed": passed,
                                       "criteria": scored["criteria"]})
        # Terminal reward push so learn/experience gets the episode outcome.
        self.push(
            event_type="reward",
            reward=score,
            terminal=True,
            reward_info={"score": score, "passed": passed,
                         "reason": reason, "task": self._task.name if self._task else ""},
        )
        self._record(browser_ep, metadata, score, obs, done=True)

        # Finalize the recording: write the terminal entry, then close the
        # artifact so it is complete even if the caller never issues `stop`.
        # A recording failure must never break termination.
        try:
            self._write_trace(action_name or reason, action_args or {}, obs,
                              done=True, ok=action_ok, error=action_error)
            if self._recorder.status().get("recording"):
                obs.metadata["record"] = self._recorder.stop()
            else:
                prior = self._recorder.status()
                if prior.get("path"):
                    obs.metadata["record"] = prior
        except Exception:
            logger.warning("recording finalize failed", exc_info=True)
        return obs

    # ── observation helpers ────────────────────────────────────────────────
    def _state_dict(self, snapshot=None) -> Dict[str, Any]:
        ep = self._ep
        base: Dict[str, Any] = {
            "url": ep.url if ep else "",
            "title": ep.title if ep else "",
            "goal": ep.goal if ep else "",
            "task": self._task.name if self._task else "",
            "step_num": ep.step_num if ep else 0,
            "max_steps": ep.max_steps if ep else self._default_max_steps,
            "visited": list(ep.visited) if ep else [],
            "history_len": len(self._actions),
            "extracted_count": len(ep.extracted) if ep else 0,
            "screenshot_count": ep.screenshot_count if ep else 0,
            "done": ep.done if ep else False,
            "success": ep.success if ep else False,
            "last_action": ep.last_action if ep else "",
            "last_error": ep.last_error if ep else "",
        }
        if snapshot is not None:
            base["page_text"] = snapshot.text
            base["elements"] = [e.to_dict() for e in snapshot.elements]
            base["truncated_text"] = snapshot.truncated_text
            base["truncated_elements"] = snapshot.truncated_elements
        return base

    def _observation(self, episode_id: str, step_num: int, *, reward: float,
                     done: bool, extra: Optional[Dict[str, Any]] = None
                     ) -> EnvObservation:
        browser_ep = self._ep
        assert browser_ep is not None
        browser_ep.step_num = step_num
        snapshot = self._last_snapshot
        state = self._state_dict(snapshot)
        render = snapshot.render_text() if snapshot is not None else ""
        return EnvObservation(
            episode_id=episode_id,
            step_num=step_num,
            state=EnvState(**state),
            done=done,
            truncated=False,
            reward=reward,
            metadata={"render": extra.pop("render", None) if extra else None,
                      "render_text": render,
                      **(extra or {})},
        )

    def _episode_record(self) -> Dict[str, Any]:
        """The host's own view of the episode — used for scoring.

        The host's record always wins over anything the agent supplies, so an
        agent cannot self-report success.
        """
        ep = self._ep
        snapshot = self._last_snapshot
        return {
            "url": ep.url if ep else "",
            "title": ep.title if ep else "",
            "text": (snapshot.text if snapshot is not None else ""),
            "actions": list(self._actions),
            "clicked_text": list(self._clicked_text),
            "filled": dict(self._filled),
            "extracted": list(ep.extracted) if ep else [],
            "step_num": ep.step_num if ep else 0,
            "max_steps": ep.max_steps if ep else self._default_max_steps,
            "visited": list(ep.visited) if ep else [],
            "done": ep.done if ep else False,
            "finished": bool(ep and ep.done),
        }

    def _record(self, browser_ep: BrowserEpisode, action: Dict[str, Any],
                reward: float, obs: EnvObservation, *, done: bool) -> None:
        """Append one transition to the experience log (never fatal)."""
        try:
            self._experiences.record(
                env_name="browser",
                episode_id=browser_ep.episode_id or "unassigned",
                step_num=obs.step_num,
                state=self._state_dict(None),
                action=action,
                reward=float(reward),
                next_state=dict(obs.state.model_dump()) if obs.state else {},
                done=done,
                success=browser_ep.success,
            )
        except Exception:
            logger.warning("experience record failed", exc_info=True)

    # ── extended primitives ────────────────────────────────────────────────
    def spaces(self) -> Dict[str, Any]:
        return {
            "action_space": {
                "navigate": {"url": "string (absolute URL)"},
                "click": {"selector": "CSS |", "index": "int from snapshot"},
                "fill": {"selector/index": "target", "text": "string"},
                "press": {"key": "string (e.g. Enter)", "selector/index": "optional"},
                "scroll": {"direction": "down|up|top|bottom", "amount": "int"},
                "back": {}, "forward": {}, "reload": {},
                "extract": {"selector": "CSS |", "index": "int"},
                "evaluate": {"js": "string"},
                "wait_for": {"selector/index": "target", "timeout_ms": "int"},
                "screenshot": {"full_page": "bool"},
                "dom": {"mode": "capture | diff (diff vs the previous capture)",
                        "max_bytes": "int"},
                "console": {"max_entries": "int", "clear": "bool"},
                "record": {"mode": "start | stop | status",
                           "include_screenshots": "bool (start only)"},
                "finish": {"answer": "string (optional)"},
            },
            "observation_space": {
                "url": "string", "title": "string", "page_text": "string",
                "elements": "list[{index,tag,role,name,text,type,value,href,disabled}]",
                "metadata.dom": "{url,title,html,stylesheets[],interactive[],"
                                "checksum,diff?,truncated}",
                "metadata.render": "{encoding:'base64',mime:'image/png',data}"
                                   " (screenshot)",
                "metadata.record": "{path,entries,bytes,recording}",
                "metadata.console": "{new:[{type,text,ts}],total,truncated?}",
                "reward": "float (score at terminal, shaping otherwise)",
                "done": "bool",
            },
            "state_schema": {
                "url": "string", "title": "string", "goal": "string",
                "task": "string", "step_num": "int", "max_steps": "int",
                "visited": "list[string]", "done": "bool", "success": "bool",
                "last_action": "string", "last_error": "string",
            },
        }

    def render(self, mode: str = "text") -> Any:
        ep = self._require_episode()
        browser_ep = self._ep
        if mode == "screenshot":
            res = self._backend.screenshot(
                full_page=False, max_bytes=self._screenshot_max_bytes)
            return res.value if res.ok else {"error": res.error}
        if mode == "json":
            return self._state_dict(self._last_snapshot)
        if mode == "html":
            res = self._backend.evaluate("document.documentElement.outerHTML")
            return res.value if res.ok else ""
        if browser_ep is not None and self._last_snapshot is not None:
            return self._last_snapshot.render_text()
        return super().render(mode)

    def plan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Affordance hints derived from the CURRENT snapshot (not guessed)."""
        suggestions: List[Dict[str, Any]] = []
        snap = self._last_snapshot
        if snap is not None:
            for e in snap.elements[:12]:
                if e.tag == "a" and e.href:
                    suggestions.append({"action": "navigate", "url": e.href})
                elif e.tag in ("input", "textarea"):
                    suggestions.append({"action": "fill", "index": e.index,
                                        "text": "<value>"})
                else:
                    suggestions.append({"action": "click", "index": e.index})
        suggestions.append({"action": "finish"})
        return suggestions

    def list_experiences(self, episode_id: str = "",
                         limit: int = 200) -> List[Dict[str, Any]]:
        return self._experiences.list(episode_id, limit)

    def handle(self, msg: A2EMessage):
        # The SDK's EnvPlugin.supported_messages() advertises env/exp/list/* but
        # its handle() has NO branch for it — so the executor routes the message
        # here and the base returns invalid_message. This is an SDK message
        # (env namespace), so implement it (same as xa-agent-env's XceedEnvPlugin).
        if isinstance(msg, EnvExpListRequest):
            try:
                rows = self.list_experiences(msg.episode_id, msg.limit)
                return EnvExpListResponse(req_id=msg.id, episodes=rows)
            except Exception as error:  # pragma: no cover - defensive
                return A2EError(req_id=msg.id, code=EnvErrorCode.RUNTIME_ERROR,
                                message=str(error), retryable=False)

        return super().handle(msg)


__all__ = ["BrowserEnvPlugin"]