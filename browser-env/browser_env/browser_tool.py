"""BrowserToolPlugin — the SAME browser, exposed as A2E ``tools``.

Why both an env plugin and a tool plugin over one browser? They are two
different contracts, and a browser is genuinely both:

* ``env/*``  — the episodic RL surface. reset(task) → step(action) →
  reward/done, scored, replayable, logged as transitions. This is what a
  training loop drives.
* ``tool/*`` — the inference surface. The agent calls ``browser_navigate`` /
  ``browser_click`` / ... as ordinary tools inside a REACT loop, with no
  episode or reward semantics.

Both delegate to ONE backend and ONE episode record on the env plugin, so the
two surfaces can never disagree about where the browser is. If a tool is
called before any ``env/reset``, the plugin performs an implicit ad-hoc reset so
the tool always has a live page — no silent empty-page behavior.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from a2e.caps.tools.plugin import ToolPlugin
from a2e.caps.tools.protocol import (
    ToolDefinition,
    ToolErrorCode,
    ToolParameter,
)

from browser_env import actions as A

logger = logging.getLogger(__name__)


def _p(name: str, type_: str, desc: str, required: bool = False) -> ToolParameter:
    return ToolParameter(name=name, type=type_, description=desc, required=required)


class BrowserToolPlugin(ToolPlugin):
    """Exposes the browser action vocabulary as A2E native tools."""

    name = "browser_tools"

    def __init__(self, host_instance, config: Any):
        super().__init__(host_instance, config)
        cfg = config or {}
        # Name of the sibling env plugin in host_config.yaml. The executor keys
        # plugins by their config `name`, so this must match it exactly.
        self._env_plugin_name = str(cfg.get("ENV_PLUGIN") or "browser_env")
        self._default_url = str(cfg.get("DEFAULT_URL") or "")
        self._text_chars = int(cfg.get("MAX_TEXT_CHARS", A.DEFAULT_TEXT_CHARS))

    # ── plugin resolution ──────────────────────────────────────────────────
    def _env(self):
        host = getattr(self, "host_instance", None)
        if host is None or not hasattr(host, "get_plugin"):
            raise RuntimeError(
                "BrowserToolPlugin needs the host executor to resolve the "
                "browser env plugin; host_instance has no get_plugin()."
            )
        plugin = host.get_plugin(self._env_plugin_name)
        if plugin is None:
            raise RuntimeError(
                f"browser env plugin {self._env_plugin_name!r} is not registered "
                f"on the host; set ENV_PLUGIN to its host_config name."
            )
        return plugin

    def _ensure_episode(self, env):
        """Make sure a live episode + page exists before a tool runs."""
        if getattr(env, "_ep", None) is not None:
            return
        url = self._default_url
        if not url:
            # No episode and no configured default: refuse loudly rather than
            # driving "about:blank" and reporting false success.
            raise RuntimeError(
                "no browser episode is active and DEFAULT_URL is unset; call "
                "browser_navigate (or env/reset) with a URL first"
            )
        env.reset(seed=None, options={"start_url": url})

    # ── manifest ───────────────────────────────────────────────────────────
    def _list_tools(self) -> List[ToolDefinition]:
        tags = ["browser", "web"]
        return [
            ToolDefinition(
                name="browser_navigate",
                description="Open an absolute URL in the browser and return the page snapshot.",
                input_parameters=[_p("url", "string", "Absolute URL to open", True)],
                tags=tags + ["navigation"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_click",
                description="Click an element. Target it by CSS selector or by the "
                            "index shown in the latest snapshot.",
                input_parameters=[
                    _p("selector", "string", "CSS selector"),
                    _p("index", "integer", "Element index from the snapshot"),
                ],
                tags=tags + ["interaction"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_fill",
                description="Type text into an input/textarea. Target by selector or index.",
                input_parameters=[
                    _p("selector", "string", "CSS selector"),
                    _p("index", "integer", "Element index from the snapshot"),
                    _p("text", "string", "Text to enter", True),
                ],
                tags=tags + ["interaction"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_press",
                description="Press a keyboard key (e.g. 'Enter'), optionally on a target element.",
                input_parameters=[
                    _p("key", "string", "Key name, e.g. Enter", True),
                    _p("selector", "string", "CSS selector to focus first"),
                    _p("index", "integer", "Element index to focus first"),
                ],
                tags=tags + ["interaction"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_scroll",
                description="Scroll the page.",
                input_parameters=[
                    _p("direction", "string", "down | up | top | bottom"),
                    _p("amount", "integer", "Pixels to scroll"),
                ],
                tags=tags + ["interaction"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_back",
                description="Go back in browser history.",
                input_parameters=[], tags=tags + ["navigation"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_extract",
                description="Extract the visible text of an element (or the whole body).",
                input_parameters=[
                    _p("selector", "string", "CSS selector"),
                    _p("index", "integer", "Element index from the snapshot"),
                ],
                tags=tags + ["extraction"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_text",
                description="Return the current page's visible text and interactive-element snapshot.",
                input_parameters=[
                    _p("max_chars", "integer", "Cap on returned characters"),
                ],
                tags=tags + ["extraction"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_screenshot",
                description="Capture a PNG screenshot, returned base64-encoded.",
                input_parameters=[
                    _p("full_page", "boolean", "Capture the full scrollable page"),
                ],
                tags=tags + ["extraction"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_dom",
                description="Capture the style-flattened DOM: markup with every "
                            "element's COMPUTED style inline, plus all stylesheets "
                            "and the interactive-element index. Use mode='diff' to "
                            "compare against the previous capture instead (reports "
                            "what changed, with a checksum).",
                input_parameters=[
                    _p("mode", "string", "capture (default) | diff"),
                    _p("max_bytes", "integer", "Cap on captured markup bytes"),
                ],
                tags=tags + ["extraction", "diff"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_record",
                description="Record this episode's trajectory to a JSONL artifact "
                            "on the host (path, step, action, url, reward, and "
                            "optionally screenshot frames). The artifact is closed "
                            "automatically when the episode ends.",
                input_parameters=[
                    _p("mode", "string", "start | stop | status", True),
                    _p("include_screenshots", "boolean",
                       "Embed a base64 screenshot frame per step (start only)"),
                ],
                tags=tags + ["meta", "recording"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_console",
                description="Read the page's console output for this episode: "
                            "console.log/warn/error lines and uncaught JS errors. "
                            "Use it after an action that did nothing — the reason "
                            "is usually a page error here. Output is episode-scoped.",
                input_parameters=[
                    _p("max_entries", "integer", "Max lines to return"),
                    _p("clear", "boolean",
                       "Clear the buffer after reading (default: keep)"),
                ],
                tags=tags + ["extraction", "debug"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_wait",
                description="Wait until an element is present.",
                input_parameters=[
                    _p("selector", "string", "CSS selector"),
                    _p("index", "integer", "Element index from the snapshot"),
                    _p("timeout_ms", "integer", "Max wait in milliseconds"),
                ],
                tags=tags + ["interaction"], version="1.0.0",
            ),
            ToolDefinition(
                name="browser_score",
                description="Score the current episode against its task's success criteria.",
                input_parameters=[], tags=tags + ["meta"], version="1.0.0",
            ),
        ]

    # ── execution ──────────────────────────────────────────────────────────
    def _execute_tool(self, arguments, *rest) -> Dict[str, Any]:
        # The executor calls this as _execute_tool(tool_name, arguments).
        name = arguments
        args: Dict[str, Any] = rest[0] if rest else {}
        if not isinstance(args, dict):
            args = {}
        try:
            if name == "browser_navigate":
                return self._navigate(args)
            if name == "browser_click":
                return self._click(args)
            if name == "browser_fill":
                return self._fill(args)
            if name == "browser_press":
                return self._press(args)
            if name == "browser_scroll":
                return self._scroll(args)
            if name == "browser_back":
                return self._back(args)
            if name == "browser_extract":
                return self._extract(args)
            if name == "browser_text":
                return self._text(args)
            if name == "browser_screenshot":
                return self._screenshot(args)
            if name == "browser_dom":
                return self._dom(args)
            if name == "browser_record":
                return self._record(args)
            if name == "browser_console":
                return self._console(args)
            if name == "browser_wait":
                return self._wait(args)
            if name == "browser_score":
                return self._score(args)
            raise RuntimeError(f"Unknown tool: {name}")
        except Exception as exc:  # surface, never crash the host
            return self._fail(str(exc), ToolErrorCode.TOOL_ERROR)

    def _env_action(self, action_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run one action through the env plugin's FULL step path.

        Deliberately calls ``env.step(...)`` rather than ``env._dispatch(...)``:
        that is what makes the two surfaces genuinely agree. The tool call then
        consumes an episode step, applies the same reward, lands in the same
        experience log and trace, and produces the same console delta as an
        ``env/step`` would — a tool-driven browser and an episode-driven browser
        can never disagree about the episode.
        """
        env = self._env()
        self._ensure_episode(env)
        try:
            norm = A.normalize_action(
                {"action_type": action_name, "payload": args})
        except A.ActionError as exc:
            return self._fail(str(exc), ToolErrorCode.TOOL_ERROR)

        try:
            obs = env.step({"action_type": norm["name"], "payload": norm["args"]})
        except Exception as exc:
            # e.g. stepping an already-finished episode, or a backend crash.
            return self._fail(str(exc), ToolErrorCode.TOOL_ERROR)

        state = obs.state.model_dump()
        md = obs.metadata or {}
        if md.get("action_ok") is False:
            return self._fail(state.get("last_error") or f"{action_name} failed",
                              ToolErrorCode.TOOL_ERROR)

        data: Dict[str, Any] = {
            "action": action_name,
            "url": state.get("url", ""),
            "title": state.get("title", ""),
            "step_num": state.get("step_num"),
            "max_steps": state.get("max_steps"),
            "done": bool(obs.done),
            "reward": obs.reward,
            "detail": md.get("render_text", ""),
        }
        # Pass through whichever action-specific payload came back.
        for key in ("dom", "record", "console", "extract", "value", "score",
                    "passed", "criteria", "terminal_reason", "action_error"):
            if key in md and md[key] is not None:
                data[key] = md[key]
        if md.get("render"):
            data["screenshot"] = md["render"]
        return self._ok(data)

    # ── individual tools ───────────────────────────────────────────────────
    def _navigate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        url = args.get("url")
        if not url:
            return self._fail("browser_navigate requires 'url'",
                              ToolErrorCode.TOOL_ERROR)
        env = self._env()
        if getattr(env, "_ep", None) is None:
            env.reset(seed=None, options={"start_url": str(url)})
            return self._ok({"url": env._backend.url, "title": env._backend.title,
                             "detail": f"opened {env._backend.url}"})
        return self._env_action(A.NAVIGATE, {"url": str(url)})

    def _click(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not A.has_target(args):
            return self._fail("browser_click needs 'selector' or 'index'",
                              ToolErrorCode.TOOL_ERROR)
        return self._env_action(A.CLICK, args)

    def _fill(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not A.has_target(args):
            return self._fail("browser_fill needs 'selector' or 'index'",
                              ToolErrorCode.TOOL_ERROR)
        if args.get("text") is None:
            return self._fail("browser_fill needs 'text'",
                              ToolErrorCode.TOOL_ERROR)
        return self._env_action(A.FILL, args)

    def _press(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not args.get("key"):
            return self._fail("browser_press needs 'key'",
                              ToolErrorCode.TOOL_ERROR)
        return self._env_action(A.PRESS, args)

    def _scroll(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self._env_action(A.SCROLL, args)

    def _back(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self._env_action(A.BACK, args)

    def _extract(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not args.get("selector") and args.get("index") is None:
            args = {**args, "selector": "body"}
        return self._env_action(A.EXTRACT, args)

    def _wait(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not A.has_target(args):
            return self._fail("browser_wait needs 'selector' or 'index'",
                              ToolErrorCode.TOOL_ERROR)
        return self._env_action(A.WAIT_FOR, args)

    def _text(self, args: Dict[str, Any]) -> Dict[str, Any]:
        env = self._env()
        self._ensure_episode(env)
        max_chars = int(args.get("max_chars", self._text_chars))
        snap = env._backend.snapshot(max_chars, env._max_elements)
        env._last_snapshot = snap
        return self._ok({"url": snap.url, "title": snap.title,
                         "text": snap.text,
                         "elements": [e.to_dict() for e in snap.elements]})

    def _screenshot(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Route through the step path (like dom/console) so a screenshot also
        # consumes an episode step and lands in the experience log — the two
        # surfaces must stay in lockstep.
        return self._env_action(A.SCREENSHOT, dict(args))

    def _dom(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self._env_action(A.DOM, dict(args, mode=args.get("mode", "capture")))

    def _record(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self._env_action(A.RECORD, dict(args))

    def _console(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Read-only: goes through the env dispatcher so the explicit-read
        # bookkeeping (marking entries as seen) stays in one place.
        return self._env_action(A.CONSOLE, dict(args))

    def _score(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from browser_env.tasking import score_episode
        env = self._env()
        if getattr(env, "_ep", None) is None:
            return self._fail("no active episode to score",
                              ToolErrorCode.TOOL_ERROR)
        if getattr(env, "_task", None) is None:
            return self._fail("no task bound to the active episode",
                              ToolErrorCode.TOOL_ERROR)
        record = env._episode_record()
        hint = dict(args.get("record") or {})
        if hint:
            # Caller hints fill gaps ONLY; the host record overwrites them,
            # so a caller cannot self-report success.
            merged = dict(hint)
            merged.update(record)
            record = merged
        scored = score_episode(env._task, record)
        return self._ok({"episode_id": env._ep.episode_id,
                         "score": scored["score"], "passed": scored["passed"],
                         "criteria": scored["criteria"]})

    # ── result helpers ─────────────────────────────────────────────────────
    def _ok(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "tool_name": "browser", "data": data}

    def _fail(self, message: str, code) -> Dict[str, Any]:
        return {"success": False, "tool_name": "browser", "error": message,
                "error_code": str(code), "exit_code": 1}


__all__ = ["BrowserToolPlugin"]