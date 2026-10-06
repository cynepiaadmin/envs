"""Episode state + transition persistence for the browser environment.

Two deliberately separate concerns live here:

* ``BrowserEpisode`` — the in-memory agent-facing state for one episode: where
  the browser is, what it has seen, how many actions it has taken, and whether
  the goal has been met. This is what ``env/observe`` returns.
* ``ExperienceStore`` — the append-only ``(state, action, reward, next_state,
  done)`` transition log an RL loop reads. SQLite-backed so it survives the
  process and is queryable from outside the host container.

The episode lifecycle (``BrowserEpisode``) and the transition log
(``ExperienceStore``) are joined only by ``episode_id`` — never by shared
mutable state — so replaying/reading experiences can never corrupt a live
episode.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BrowserEpisode:
    """Agent-facing state of one browser episode."""

    episode_id: str
    goal: str = ""
    start_url: str = ""
    url: str = ""
    title: str = ""
    step_num: int = 0
    max_steps: int = 15
    history: List[str] = field(default_factory=list)
    visited: List[str] = field(default_factory=list)
    extracted: List[str] = field(default_factory=list)
    screenshot_count: int = 0
    done: bool = False
    success: bool = False
    last_action: str = ""
    last_error: str = ""
    created_at: float = field(default_factory=time.time)

    def to_state_dict(self) -> Dict[str, Any]:
        """Compact state surfaced in EnvObservation.state."""
        return {
            "url": self.url,
            "title": self.title,
            "goal": self.goal,
            "start_url": self.start_url,
            "step_num": self.step_num,
            "max_steps": self.max_steps,
            "history_len": len(self.history),
            "visited": list(self.visited),
            "extracted_count": len(self.extracted),
            "screenshot_count": self.screenshot_count,
            "done": self.done,
            "success": self.success,
            "last_action": self.last_action,
            "last_error": self.last_error,
        }


class ExperienceStore:
    """SQLite-backed append-only transition log, keyed by episode_id."""

    def __init__(self, root: str) -> None:
        self._root = root
        os.makedirs(self._root, exist_ok=True)
        self._db = os.path.join(self._root, "experiences.db")
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                env_name TEXT, episode_id TEXT, step_num INTEGER,
                state TEXT, action TEXT, reward REAL,
                next_state TEXT, done INTEGER, success INTEGER, ts REAL)"""
        )
        self._conn.commit()

    def record(self, *, env_name: str, episode_id: str, step_num: int,
               state: Dict[str, Any], action: Dict[str, Any], reward: float,
               next_state: Dict[str, Any], done: bool, success: bool) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO experiences (env_name, episode_id, step_num, state, "
                "action, reward, next_state, done, success, ts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (env_name, episode_id, step_num,
                 json.dumps(state, default=str), json.dumps(action, default=str),
                 float(reward), json.dumps(next_state, default=str),
                 int(bool(done)), int(bool(success)), time.time()),
            )
            self._conn.commit()

    def list(self, episode_id: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            if episode_id:
                rows = self._conn.execute(
                    "SELECT episode_id,step_num,state,action,reward,next_state,"
                    "done,success FROM experiences WHERE episode_id=? "
                    "ORDER BY id ASC LIMIT ?", (episode_id, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT episode_id,step_num,state,action,reward,next_state,"
                    "done,success FROM experiences ORDER BY id DESC LIMIT ?",
                    (limit,)).fetchall()
        out = []
        for r in rows:
            out.append({
                "episode_id": r[0], "step_num": r[1],
                "state": _loads(r[2]), "action": _loads(r[3]),
                "reward": r[4], "next_state": _loads(r[5]),
                "done": bool(r[6]), "success": bool(r[7]),
            })
        return out

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


def _loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


class EpisodeRecorder:
    """Append-only JSONL recording of one episode's trajectory.

    Why JSONL and not Playwright's video: a recording must work identically for
    the ``playwright`` and ``scripted`` backends, and it must be readable by
    tests, graders and RL analysis without a player. Each line is one step
    (action, url, reward, done, and optionally a base64 screenshot frame).

    Lifecycle: ``start()`` → ``record(...)`` per step → ``stop()``. All three are
    no-ops when recording is not active, so the env can call ``record`` on every
    step unconditionally. A failure to write must never break an episode — the
    recorder degrades to in-memory counting and reports it on ``stop()``.

    This is deliberately separate from ``ExperienceStore``: that store is the
    RL ``(s, a, r, s', done)`` log (SQLite, for training), while this is the
    human/debug replay artifact (JSONL, for inspection). Different consumers,
    different formats, different lifetimes.
    """

    def __init__(self, root: str) -> None:
        self._dir = os.path.join(root, "recordings")
        self._lock = threading.RLock()
        self._active = False
        self._episode_id = ""
        self._path = ""
        self._frames = False
        self._entries = 0
        self._bytes = 0
        self._started_at = 0.0
        self._errors: List[str] = []

    def start(self, episode_id: str, include_screenshots: bool = False,
              root: str = "") -> Dict[str, Any]:
        """Begin recording. Starting twice restarts (a new file)."""
        with self._lock:
            base_dir = os.path.join(root, "recordings") if root else self._dir
            try:
                os.makedirs(base_dir, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                self._path = os.path.join(
                    base_dir, f"{stamp}-{episode_id or 'episode'}.jsonl")
                self._fh = open(self._path, "w", encoding="utf-8")
            except OSError as exc:
                self._path = ""
                self._errors.append(f"open failed: {exc}")
            self._active = True
            self._episode_id = episode_id
            self._frames = bool(include_screenshots)
            self._entries = 0
            self._bytes = 0
            self._started_at = time.time()
            return {"recording": True, "path": self._path,
                    "include_screenshots": self._frames,
                    "errors": list(self._errors)}

    def record(self, entry: Dict[str, Any]) -> None:
        """Append one entry. No-op (never raises) when not recording."""
        with self._lock:
            if not self._active:
                return
            self._entries += 1
            line = ""
            try:
                line = json.dumps(entry, default=str, separators=(",", ":")) + "\n"
                self._bytes += len(line)
                fh = getattr(self, "_fh", None)
                if fh is not None:
                    fh.write(line)
                    fh.flush()
            except (OSError, TypeError, ValueError) as exc:
                self._errors.append(f"write failed at entry {self._entries}: {exc}")

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"recording": self._active, "path": self._path,
                    "entries": self._entries, "bytes": self._bytes,
                    "include_screenshots": self._frames,
                    "episode_id": self._episode_id,
                    "elapsed_s": (time.time() - self._started_at)
                    if self._started_at else 0.0,
                    "errors": list(self._errors[-5:])}

    def stop(self) -> Dict[str, Any]:
        """Close the file and report the artifact. Idempotent."""
        with self._lock:
            if not self._active:
                return dict(self.status(), detail="not recording")
            try:
                fh = getattr(self, "_fh", None)
                if fh is not None:
                    fh.close()
            except OSError as exc:
                self._errors.append(f"close failed: {exc}")
            self._active = False
            out = self.status()
            out["detail"] = "stopped"
            return out


__all__ = ["BrowserEpisode", "ExperienceStore", "EpisodeRecorder"]