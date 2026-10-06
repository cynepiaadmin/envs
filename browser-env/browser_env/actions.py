"""Flat action vocabulary shared by the browser host plugins.

This module is deliberately dependency-free (stdlib only) so it can be imported
by the env plugin, the tool plugin, the backends, and the tests without pulling
in pydantic/playwright.

The browser agent repo ships its OWN copy of the tool schemas it advertises to
the LLM (browser_agent/runtime/browser_tools.py). The two repos must never
import each other — they agree on the wire contract below, nothing more.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ── Action names ────────────────────────────────────────────────────────────
NAVIGATE = "navigate"
CLICK = "click"
FILL = "fill"
PRESS = "press"
SCROLL = "scroll"
BACK = "back"
FORWARD = "forward"
RELOAD = "reload"
EXTRACT = "extract"
DOM = "dom"
EVALUATE = "evaluate"
WAIT_FOR = "wait_for"
SCREENSHOT = "screenshot"
CONSOLE = "console"
RECORD = "record"
FINISH = "finish"

#: Every action the environment accepts. Anything else → UNKNOWN_ACTION.
ACTION_NAMES: List[str] = [
    NAVIGATE, CLICK, FILL, PRESS, SCROLL, BACK, FORWARD, RELOAD,
    EXTRACT, DOM, EVALUATE, WAIT_FOR, SCREENSHOT, CONSOLE, RECORD, FINISH,
]

#: Actions that mutate page state (recorded as "effects" in the trace).
MUTATING: set[str] = {NAVIGATE, CLICK, FILL, PRESS, BACK, FORWARD, RELOAD, WAIT_FOR}

#: Actions that are read-only observations of the page.
READ_ONLY: set[str] = {EXTRACT, DOM, EVALUATE, SCREENSHOT, CONSOLE}

#: Actions that end the episode.
TERMINAL: set[str] = {FINISH}

# ── Bounds (no silent defaults: every one of these is an explicit contract) ─
DEFAULT_SCROLL_AMOUNT = 500
DEFAULT_WAIT_MS = 5000
DEFAULT_TEXT_CHARS = 4000
DEFAULT_MAX_ELEMENTS = 120
DEFAULT_SCREENSHOT_MAX_BYTES = 1_500_000
DEFAULT_DOM_MAX_BYTES = 500_000


class ActionError(ValueError):
    """Raised when an action payload is malformed or unsupported."""


def normalize_action(action: Any) -> Dict[str, Any]:
    """Return a canonical ``{"name": ..., "args": {...}}`` dict.

    Accepts the wire shapes the protocol can deliver:

      * ``{"action_type": "click", "payload": {...}}``   (EnvAction model_dump)
      * ``{"type": "click", "args": {...}}``             (loose / OpenAI style)
      * ``{"name": "click", "arguments": {...}}``        (tool-call style)
      * ``{"name": "click", ...kwargs}``                 (flat kwargs)

    Raises ActionError on anything that carries no recognisable action name.
    """
    if action is None:
        raise ActionError("empty action")

    if not isinstance(action, dict):
        # EnvAction is a pydantic model on the wire; accept it structurally.
        action = {
            "action_type": getattr(action, "action_type", None),
            "payload": getattr(action, "payload", None) or {},
        }

    name = (
        action.get("action_type")
        or action.get("name")
        or action.get("type")
        or action.get("tool")
    )
    if not name:
        raise ActionError(f"action has no name: {sorted(action.keys())}")
    name = str(name).strip().lower()

    args = action.get("payload")
    if args is None:
        args = action.get("args")
    if args is None:
        args = action.get("arguments")
    if args is None:
        # flat form — every other key is an argument
        reserved = {"action_type", "name", "type", "tool"}
        args = {k: v for k, v in action.items() if k not in reserved}
    if not isinstance(args, dict):
        raise ActionError(f"action args must be an object, got {type(args).__name__}")

    return {"name": name, "args": args}


def default_selector_selector(action_name: str) -> str:
    return f"{action_name}: either 'selector' (CSS) or 'index' (from the snapshot) is required"


def has_target(args: Dict[str, Any]) -> bool:
    """True when the args carry a usable element target (selector or index)."""
    return bool(args.get("selector")) or args.get("index") is not None


def resolve_target(args: Dict[str, Any], action_name: str) -> str:
    """Return a CSS selector for the action's target, or raise ActionError."""
    sel = args.get("selector")
    if sel:
        return str(sel)
    idx = args.get("index")
    if idx is None:
        raise ActionError(default_selector_selector(action_name))
    try:
        idx = int(idx)
    except (TypeError, ValueError) as exc:
        raise ActionError(f"index must be an integer, got {idx!r}") from exc
    # The snapshot tags every interactive element with data-a2e-eid, so an index
    # is a stable, replayable target — not a positional guess.
    return f'[data-a2e-eid="{idx}"]'