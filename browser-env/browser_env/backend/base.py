"""Browser backend abstraction (the env's "provider layer").

The env plugin depends ONLY on the ``BrowserBackend`` contract defined here.
A real Playwright backend and a deterministic scripted backend both implement
it, so the whole environment is testable offline while still driving a real
browser in production. This mirrors the xa-agent provider-independence rule:
the consumer (the env plugin) never imports a concrete driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ElementInfo:
    """One interactive element exposed in a page snapshot."""

    index: int
    tag: str
    role: str = ""
    name: str = ""
    text: str = ""
    type: str = ""
    value: str = ""
    href: str = ""
    disabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "tag": self.tag,
            "role": self.role,
            "name": self.name,
            "text": self.text,
            "type": self.type,
            "value": self.value,
            "href": self.href,
            "disabled": self.disabled,
        }

    def label(self) -> str:
        bits = [self.tag]
        if self.type:
            bits.append(f"[{self.type}]")
        if self.name:
            bits.append(f"name={self.name}")
        shown = self.text or self.value
        if shown:
            bits.append(f'"{shown[:60]}"')
        if self.disabled:
            bits.append("(disabled)")
        return " ".join(bits)


@dataclass
class PageSnapshot:
    """A compact, LLM-friendly view of the current page."""

    url: str
    title: str
    text: str = ""
    elements: List[ElementInfo] = field(default_factory=list)
    truncated_text: bool = False
    truncated_elements: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "truncated_text": self.truncated_text,
            "truncated_elements": self.truncated_elements,
            "elements": [e.to_dict() for e in self.elements],
        }

    def render_text(self) -> str:
        """Human/LLM-readable rendering used as the env observation body."""
        lines = [f"URL: {self.url}", f"TITLE: {self.title}"]
        if self.elements:
            lines.append("INTERACTIVE ELEMENTS:")
            for e in self.elements:
                lines.append(f"  [{e.index}] {e.label()}")
        if self.text:
            lines.append("PAGE TEXT:")
            lines.append(self.text)
        return "\n".join(lines)


@dataclass
class ActionResult:
    """Outcome of one backend action."""

    ok: bool
    action: str
    detail: str = ""
    error: str = ""
    value: Any = None
    navigated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "detail": self.detail,
            "error": self.error,
            "value": self.value,
            "navigated": self.navigated,
        }


class BrowserBackend:
    """Contract every browser driver implements.

    Lifecycle: ``start()`` once per host process, ``close()`` on teardown.
    An episode is a *navigation session*: ``reset(url)`` puts the browser on a
    known page and clears per-episode state; it never restarts the browser.
    """

    name = "abstract"

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def reset_session(self, url: str, timeout_ms: int = 15000) -> ActionResult:
        """Start a fresh episode at ``url`` (clear cookies/state where cheap)."""
        raise NotImplementedError

    # ── state ──────────────────────────────────────────────────────────────
    @property
    def url(self) -> str:
        raise NotImplementedError

    @property
    def title(self) -> str:
        raise NotImplementedError

    def snapshot(
        self,
        max_text_chars: int = 4000,
        max_elements: int = 120,
    ) -> PageSnapshot:
        raise NotImplementedError

    # ── actions ────────────────────────────────────────────────────────────
    def navigate(self, url: str, timeout_ms: int = 15000) -> ActionResult:
        raise NotImplementedError

    def click(self, selector: str, timeout_ms: int = 10000) -> ActionResult:
        raise NotImplementedError

    def fill(self, selector: str, text: str, timeout_ms: int = 10000) -> ActionResult:
        raise NotImplementedError

    def press(self, key: str, selector: str = "") -> ActionResult:
        raise NotImplementedError

    def scroll(self, direction: str = "down", amount: int = 500) -> ActionResult:
        raise NotImplementedError

    def back(self) -> ActionResult:
        raise NotImplementedError

    def forward(self) -> ActionResult:
        raise NotImplementedError

    def reload(self) -> ActionResult:
        raise NotImplementedError

    def extract(self, selector: str) -> ActionResult:
        raise NotImplementedError

    def evaluate(self, js: str) -> ActionResult:
        raise NotImplementedError

    def wait_for(self, selector: str, timeout_ms: int = 5000) -> ActionResult:
        raise NotImplementedError

    def screenshot(
        self,
        full_page: bool = False,
        max_bytes: int = 1_500_000,
    ) -> ActionResult:
        raise NotImplementedError

    def html(self, max_bytes: int = 500_000) -> ActionResult:
        """DOM capture: the serialized outer HTML of the current document.

        Kept separate from ``screenshot`` (pixels) and ``extract`` (visible text)
        because they answer different questions — a screenshot is only useful to
        a vision model, while the DOM is what a non-vision agent needs to find
        selectors the element snapshot may have missed.
        """
        raise NotImplementedError

    def console(self, max_entries: int = 200, clear: bool = False) -> ActionResult:
        """Browser console output for the current episode.

        Captures ``console.*`` messages and uncaught page errors — the two
        things that tell you WHY an action did nothing (a JS exception, a
        blocked resource, a 4xx/5xx logged by the page). Scoped to the episode:
        ``reset_session`` clears it so captures are not mixed across episodes.

        Returns ``{"entries": [{type, text, ts, ...}], "count": int}``.
        """
        raise NotImplementedError


def make_backend(kind: str, **kwargs) -> BrowserBackend:
    """Factory. Only ``scripted`` and ``playwright`` are supported.

    Concrete drivers are imported LAZILY so importing this module never
    requires playwright to be installed (the scripted backend and the env
    plugin tests run with zero browser dependency).
    """
    kind = (kind or "").strip().lower()
    if kind == "scripted":
        from browser_env.backend.scripted_backend import ScriptedBrowser
        return ScriptedBrowser(**kwargs)
    if kind == "playwright":
        from browser_env.backend.playwright_backend import PlaywrightBrowser
        return PlaywrightBrowser(**kwargs)
    raise ValueError(
        f"unknown browser backend {kind!r}; expected 'playwright' or 'scripted'"
    )


__all__ = [
    "BrowserBackend",
    "ElementInfo",
    "PageSnapshot",
    "ActionResult",
    "make_backend",
]