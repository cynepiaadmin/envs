"""Deterministic, dependency-free browser backend for tests and offline runs.

It models a tiny site graph in memory and implements the same ``BrowserBackend``
contract as the Playwright driver. Nothing about it is a mock of Playwright —
it is a genuine, self-contained browser model, so the env plugin, the tool
plugin, and the agent can all be exercised end to end with no network and no
browser installed.

Default site graph (overridable via ``pages=``):

    https://shop.test/                 home — 2 products, search box
    https://shop.test/product/1        product page — "Add to cart"
    https://shop.test/cart             cart — shows the added items
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from browser_env.backend.base import (
    ActionResult,
    BrowserBackend,
    ElementInfo,
    PageSnapshot,
)


@dataclass
class FakePage:
    url: str
    title: str
    text: str
    elements: List[ElementInfo] = field(default_factory=list)
    #: numeric element index -> (action, value) the element performs
    handlers: Dict[int, tuple] = field(default_factory=dict)
    #: console lines this page logs on load (authored, deterministic)
    console: List[str] = field(default_factory=list)


def _el(idx: int, tag: str, text: str, **kw) -> ElementInfo:
    return ElementInfo(index=idx, tag=tag, text=text, **kw)


def default_pages() -> Dict[str, FakePage]:
    home = FakePage(
        url="https://shop.test/",
        title="Shop Home",
        text="Welcome to Shop. Featured products: Widget, Gadget.",
        elements=[
            _el(0, "input", "", name="q", type="search", value=""),
            _el(1, "a", "Widget", href="https://shop.test/product/1"),
            _el(2, "a", "Gadget", href="https://shop.test/product/2"),
            _el(3, "a", "Cart", href="https://shop.test/cart"),
        ],
        handlers={
            0: ("fill", "q"),
            1: ("navigate", "https://shop.test/product/1"),
            2: ("navigate", "https://shop.test/product/2"),
            3: ("navigate", "https://shop.test/cart"),
        },
    )
    product = FakePage(
        url="https://shop.test/product/1",
        title="Widget — Product",
        text="Widget. Price: $9.99. The widget is a small round thing.",
        elements=[
            _el(0, "button", "Add to cart", name="add"),
            _el(1, "a", "Back to home", href="https://shop.test/"),
        ],
        handlers={
            0: ("add_to_cart", "Widget"),
            1: ("navigate", "https://shop.test/"),
        },
    )
    gadget = FakePage(
        url="https://shop.test/product/2",
        title="Gadget — Product",
        text="Gadget. Price: $19.99. The gadget is a large square thing.",
        elements=[
            _el(0, "button", "Add to cart", name="add"),
            _el(1, "a", "Back to home", href="https://shop.test/"),
        ],
        handlers={
            0: ("add_to_cart", "Gadget"),
            1: ("navigate", "https://shop.test/"),
        },
    )
    cart = FakePage(
        url="https://shop.test/cart",
        title="Your Cart",
        text="Your cart is empty.",
        elements=[_el(0, "a", "Back to home", href="https://shop.test/")],
        handlers={0: ("navigate", "https://shop.test/")},
        # Authored console output: a real cart page logs a fallback warning
        # here, and an agent chasing "why is my cart empty?" should see it.
        console=["warn: localStorage unavailable, falling back to memory cart",
                 "cart: rendered 0 line items"],
    )
    return {p.url: p for p in (home, product, gadget, cart)}


# ── the scripted page's stylesheet + computed-style table ─────────────────
# Deterministic on purpose: a DOM capture of an unchanged page must always
# produce the same checksum, and the capture must carry real style content so
# a diff of two captures is meaningful (same shape as Chromium's output).
_STYLESHEET_CSS = (
    "h1 { font-size: 24px; font-weight: 700; color: #111111; margin: 0 0 8px 0; }\n"
    "p  { font-size: 14px; color: #333333; line-height: 1.5; margin: 0 0 12px 0; }\n"
    "a  { color: #0066cc; text-decoration: underline; font-size: 14px; }\n"
    "button { background-color: #f2f2f2; border-top-width: 1px; "
    "border-top-color: #cccccc; padding: 4px 8px; font-size: 14px; "
    "cursor: pointer; }\n"
    "input { border-top-width: 1px; border-top-color: #999999; "
    "padding: 4px; font-size: 14px; }\n"
)

#: Flattened computed styles, keyed by tag — the analogue of Chromium's
#: getComputedStyle output written inline on the cloned element.
_COMPUTED_STYLE = {
    "h1": "display:block;color:#111111;font-size:24px;font-weight:700;"
          "margin-bottom:8px;line-height:1.2;",
    "p": "display:block;color:#333333;font-size:14px;line-height:1.5;"
         "margin-bottom:12px;",
    "a": "display:inline;color:#0066cc;text-decoration:underline;font-size:14px;"
         "cursor:pointer;",
    "button": "display:inline-block;background-color:#f2f2f2;"
              "border-top-width:1px;border-top-color:#cccccc;"
              "padding-top:4px;padding-left:8px;font-size:14px;cursor:pointer;",
    "input": "display:inline-block;border-top-width:1px;"
             "border-top-color:#999999;padding-top:4px;font-size:14px;",
    "option": "display:block;font-size:14px;",
    "textarea": "display:block;border-top-width:1px;border-top-color:#999999;"
                "font-size:14px;",
}


class ScriptedBrowser(BrowserBackend):
    """In-memory browser model implementing the full backend contract."""

    name = "scripted"
    def __init__(self, pages: Optional[Dict[str, FakePage]] = None,
                 start_url: str = "https://shop.test/",
                 search_results: Optional[Dict[str, str]] = None,
                 **_ignored: Any) -> None:
        self._pages = pages or default_pages()
        self._start_url = start_url
        self._search = {
            "widget": "https://shop.test/product/1",
            "gadget": "https://shop.test/product/2",
        }
        if search_results:
            self._search.update(search_results)
        self._history: List[str] = []
        self._cursor = -1
        self._cart: List[str] = []
        self._started = False
        self._current = self._start_url
        self._inputs: Dict[str, str] = {}
        #: console buffer for the current episode (see console())
        self._console: List[Dict[str, Any]] = []

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        self._started = True

    def close(self) -> None:
        self._started = False

    # ── console capture (episode-scoped, mirrors the Playwright backend) ───
    def _log(self, type_: str, text: str) -> None:
        self._console.append({"type": type_, "text": text, "args": [],
                              "ts": float(len(self._console))})

    def _load_console(self, page: FakePage) -> None:
        """Emit what a page logs on load: the load line + authored lines."""
        self._log("log", f"page loaded: {page.url}")
        for line in page.console:
            if ":" in line and line.split(":", 1)[0].strip() in (
                    "warn", "warning", "error", "info", "log", "debug"):
                kind, _, rest = line.partition(":")
                kind = {"warning": "warn"}.get(kind.strip(), kind.strip())
                self._log(kind, rest.strip())
            else:
                self._log("log", line)

    def console(self, max_entries: int = 200, clear: bool = False) -> ActionResult:
        entries = list(self._console[-max_entries:])
        total = len(self._console)
        if clear:
            self._console.clear()
        return ActionResult(
            True, "console",
            detail=f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}",
            value={"entries": entries, "count": len(entries), "total": total,
                   "dropped": 0, "cleared": clear},
        )

    def reset_session(self, url: str, timeout_ms: int = 15000) -> ActionResult:
        if url not in self._pages:
            return ActionResult(False, "reset", error=f"no such page: {url}")
        self._cart.clear()
        self._inputs.clear()
        self._history = [url]
        self._cursor = 0
        self._current = url
        # New episode: clean console (same rule as the Playwright backend).
        self._console.clear()
        self._load_console(self._pages[url])
        return ActionResult(True, "reset", detail=f"opened {url}", navigated=True)

    # ── state ──────────────────────────────────────────────────────────────
    @property
    def url(self) -> str:
        return self._current

    @property
    def title(self) -> str:
        page = self._pages.get(self._current)
        return page.title if page else ""

    def _page(self) -> FakePage:
        page = self._pages.get(self._current)
        if page is None:
            raise RuntimeError(f"no such page: {self._current}")
        if page.url.endswith("/cart"):
            items = ", ".join(self._cart) if self._cart else ""
            text = (f"Your cart contains: {items}." if items
                    else "Your cart is empty.")
            page = FakePage(url=page.url, title=page.title, text=text,
                            elements=page.elements, handlers=page.handlers)
        return page

    def snapshot(self, max_text_chars: int = 4000,
                 max_elements: int = 120) -> PageSnapshot:
        page = self._page()
        els = page.elements[:max_elements]
        text = page.text
        truncated = len(text) > max_text_chars
        return PageSnapshot(
            url=page.url, title=page.title, text=text[:max_text_chars],
            elements=list(els), truncated_text=truncated,
            truncated_elements=len(page.elements) > max_elements,
        )

    # ── actions ────────────────────────────────────────────────────────────
    def navigate(self, url: str, timeout_ms: int = 15000) -> ActionResult:
        if url not in self._pages:
            return ActionResult(False, "navigate", error=f"no such page: {url}")
        self._current = url
        self._history = self._history[: self._cursor + 1] + [url]
        self._cursor = len(self._history) - 1
        # Navigation re-runs the page, so it logs again (like a real reload).
        self._load_console(self._pages[url])
        return ActionResult(True, "navigate", detail=f"opened {url}",
                            navigated=True)

    def _resolve(self, selector: str) -> tuple:
        m = re.fullmatch(r'\[data-a2e-eid="(\d+)"\]', selector.strip())
        if not m:
            return None, ActionResult(
                False, "click",
                error=f"scripted backend only resolves data-a2e-eid selectors, got {selector!r}",
            )
        idx = int(m.group(1))
        return idx, None

    def click(self, selector: str, timeout_ms: int = 10000) -> ActionResult:
        page = self._page()
        idx, err = self._resolve(selector)
        if err:
            return err
        el = next((e for e in page.elements if e.index == idx), None)
        if el is None:
            return ActionResult(False, "click", error=f"no element {idx}")
        if el.disabled:
            return ActionResult(False, "click", error=f"element {idx} disabled")
        handler = page.handlers.get(idx)
        if handler is None:
            return ActionResult(True, "click", detail=f"clicked {idx} (no-op)")
        kind, value = handler
        if kind == "navigate":
            return self.navigate(value)
        if kind == "add_to_cart":
            if value not in self._cart:
                self._cart.append(value)
            self._log("log", f"cart: added {value} (total={len(self._cart)})")
            return ActionResult(True, "click", detail=f"added {value} to cart")
        return ActionResult(True, "click", detail=f"clicked {idx}")

    def fill(self, selector: str, text: str, timeout_ms: int = 10000) -> ActionResult:
        page = self._page()
        idx, err = self._resolve(selector)
        if err:
            return err
        el = next((e for e in page.elements if e.index == idx), None)
        if el is None:
            return ActionResult(False, "fill", error=f"no element {idx}")
        if el.tag not in ("input", "textarea", "select"):
            return ActionResult(False, "fill",
                                error=f"element {idx} is not fillable ({el.tag})")
        el.value = text
        self._inputs[el.name or f"eid{idx}"] = text
        return ActionResult(True, "fill", detail=f"filled {idx}", value=text)

    def press(self, key: str, selector: str = "") -> ActionResult:
        key = (key or "").strip()
        if key in ("Enter", "NumpadEnter"):
            q = (self._inputs.get("q") or "").strip().lower()
            target = self._search.get(q)
            if target:
                return self.navigate(target)
            return ActionResult(True, "press",
                                detail=f"pressed Enter with q={q!r} (no results)")
        return ActionResult(True, "press", detail=f"pressed {key}")

    def scroll(self, direction: str = "down", amount: int = 500) -> ActionResult:
        return ActionResult(True, "scroll", detail=f"scrolled {direction}")

    def back(self) -> ActionResult:
        if self._cursor > 0:
            self._cursor -= 1
            self._current = self._history[self._cursor]
            return ActionResult(True, "back", detail=f"back to {self._current}",
                                navigated=True)
        return ActionResult(True, "back", detail="no earlier page")

    def forward(self) -> ActionResult:
        if self._cursor < len(self._history) - 1:
            self._cursor += 1
            self._current = self._history[self._cursor]
            return ActionResult(True, "forward", detail=f"forward to {self._current}",
                                navigated=True)
        return ActionResult(True, "forward", detail="no later page")

    def reload(self) -> ActionResult:
        return ActionResult(True, "reload", detail="reloaded", navigated=True)

    def extract(self, selector: str) -> ActionResult:
        page = self._page()
        if not selector or selector in ("body", "html"):
            return ActionResult(True, "extract", detail="extracted body",
                                value=page.text)
        idx, err = self._resolve(selector)
        if err:
            return err
        el = next((e for e in page.elements if e.index == idx), None)
        if el is None:
            return ActionResult(False, "extract", error=f"no element {idx}")
        return ActionResult(True, "extract", detail=f"extracted {idx}",
                            value=el.text or el.value)

    def evaluate(self, js: str) -> ActionResult:
        page = self._page()
        if "document.title" in js:
            return ActionResult(True, "evaluate", value=page.title)
        if "location.href" in js or "document.URL" in js:
            return ActionResult(True, "evaluate", value=page.url)
        return ActionResult(True, "evaluate",
                            value={"note": "scripted backend only supports "
                                           "document.title / location.href"})

    def wait_for(self, selector: str, timeout_ms: int = 5000) -> ActionResult:
        idx, err = self._resolve(selector)
        if err:
            return err
        page = self._page()
        found = any(e.index == idx for e in page.elements)
        if found:
            return ActionResult(True, "wait_for", detail=f"found {idx}")
        return ActionResult(False, "wait_for", error=f"element {idx} not present")

    def screenshot(self, full_page: bool = False,
                   max_bytes: int = 1_500_000) -> ActionResult:
        # 1x1 transparent PNG — a real, decodable image, no binary fixture file.
        png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
               b"\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc"
               b"\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
        import base64
        return ActionResult(
            True, "screenshot", detail=f"{len(png)} bytes",
            value={"encoding": "base64", "mime": "image/png",
                   "bytes": len(png),
                   "data": base64.b64encode(png).decode("ascii"),
                   "omitted": False},
        )

    def html(self, max_bytes: int = 500_000) -> ActionResult:
        """Style-flattened DOM capture matching the Playwright backend's shape.

        The scripted backend must produce the SAME diff-relevant structure as a
        real browser — markup with computed styles inline, the stylesheet text,
        and the interactive-element index — so a diff written against the
        scripted backend also works against Chromium (and vice versa).

        Everything here is derived from the same FakePage the snapshot reads, so
        the DOM a test diffs and the elements the agent clicks are one source of
        truth. Nothing is randomised: two captures of an unchanged page have an
        identical checksum, and a changed page changes it.
        """
        page = self._page()
        rows = []
        interactive = []
        for e in page.elements:
            attrs = []
            if e.name:
                attrs.append(f'name="{e.name}"')
            if e.type:
                attrs.append(f'type="{e.type}"')
            if e.href:
                attrs.append(f'href="{e.href}"')
            if e.value:
                attrs.append(f'value="{e.value}"')
            attrs.append(f'data-a2e-eid="{e.index}"')
            style = _COMPUTED_STYLE.get(e.tag, "display:block;font-size:14px;")
            attr_str = " " + " ".join(attrs)
            style_attr = f' style="{style}"'
            if e.tag == "input":
                rows.append(f"<input{attr_str}{style_attr}>")
            else:
                rows.append(f"<{e.tag}{attr_str}{style_attr}>{e.text}</{e.tag}>")
            interactive.append({"index": e.index, "tag": e.tag,
                                "name": e.name, "href": e.href,
                                "text": e.text[:120]})

        doc = (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            f"<title>{page.title}</title>\n"
            f"<style>{_STYLESHEET_CSS}</style>\n"
            "</head>\n<body>\n"
            f"<h1 style=\"{_COMPUTED_STYLE['h1']}\">{page.title}</h1>\n"
            f"<p style=\"{_COMPUTED_STYLE['p']}\">{page.text}</p>\n"
            + "\n".join(rows)
            + "\n</body>\n</html>\n"
        )
        over = len(doc) > max_bytes
        if over:
            doc = doc[:max_bytes]
            cut = doc.rfind(">")
            if cut > 0:
                doc = doc[: cut + 1]
        css_text = _STYLESHEET_CSS
        return ActionResult(
            True, "dom",
            detail=f"{len(doc)} chars over 1 stylesheet(s)"
                   f"{' (truncated)' if over else ''}",
            value={
                "url": page.url,
                "title": page.title,
                "html": doc,
                "stylesheets": [{
                    "href": "(inline)",
                    "text": css_text,
                    "readable": True,
                    "rules": len(css_text.split("\n")),
                }],
                "interactive": interactive,
                "element_count": len(page.elements) + 2,
                "css_bytes": len(css_text),
                "stylesheet_count": 1,
                "truncated": over,
            },
        )


__all__ = ["ScriptedBrowser", "FakePage", "default_pages"]