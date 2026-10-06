"""Real browser backend — Playwright (sync API).

Drives a headless Chromium page. Imports playwright LAZILY so the rest of the
package (and the scripted backend) works with zero browser dependency.

Snapshot model
--------------
Before each ``snapshot()`` we tag every interactive element with a stable
``data-a2e-eid`` attribute, then return the tagged elements. The agent targets
elements either by CSS ``selector`` or by the ``index`` it saw in the snapshot,
which the env resolves back to ``[data-a2e-eid="N"]`` — so an action is
replayable and never depends on the model inventing a correct CSS path.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from browser_env.backend.base import (
    ActionResult,
    BrowserBackend,
    ElementInfo,
    PageSnapshot,
)

logger = logging.getLogger(__name__)

# Tags + roles considered "interactive" for the snapshot.
_INTERACTIVE_SELECTOR = (
    "a, button, input, textarea, select, option, [role=button], [role=link], "
    "[role=checkbox], [role=radio], [role=tab], [role=menuitem], [role=switch], "
    "[contenteditable=true], [onclick]"
)

# JS that stamps a stable index on each interactive element and returns them.
_SNAPSHOT_JS = r"""
(opts) => {
  const sel = opts.sel;
  const maxElements = opts.maxElements;
  const nodes = Array.from(document.querySelectorAll(sel));
  const visible = nodes.filter((el) => {
    const r = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    if (r.width === 0 && r.height === 0) return false;
    return true;
  });
  const out = [];
  let i = 0;
  for (const el of visible) {
    if (i >= maxElements) break;
    el.setAttribute('data-a2e-eid', String(i));
    const label = (el.getAttribute('aria-label') || el.innerText
                   || el.value || el.getAttribute('placeholder') || '');
    out.push({
      index: i,
      tag: el.tagName.toLowerCase(),
      role: el.getAttribute('role') || '',
      name: el.getAttribute('name') || '',
      text: (label || '').trim().slice(0, 120),
      type: el.getAttribute('type') || '',
      value: (el.value || '').toString().slice(0, 120),
      href: el.getAttribute('href') || '',
      disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true'),
    });
    i += 1;
  }
  return { elements: out, url: location.href, title: document.title };
}
"""

# JS used for the text-only observation body.
_TEXT_JS = r"""
(maxChars) => {
  const body = document.body ? document.body.innerText : '';
  const text = (body || '').replace(/\n{3,}/g, '\n\n').trim();
  return {
    url: location.href,
    title: document.title,
    text: text.slice(0, maxChars),
    truncated: text.length > maxChars
  };
}
"""

# JS for DOM capture. The output must be sufficient to DIFF two captures, so it
# is not a raw outerHTML dump:
#   * every stylesheet is captured (inline <style> rules AND link/externals)
#   * every element is flattened with its COMPUTED style written inline, so two
#     captures differ if ANY resolved property differs — even when only an
#     external stylesheet changed
#   * interactive elements keep their data-a2e-eid so element indexes survive
#   * the live DOM is never mutated: we clone, annotate the clone, serialize it
_DOM_JS = r"""
(opts) => {
  const maxBytes = opts.maxBytes;
  const sel = opts.sel;
  const PROPS = [
    'display','visibility','position','top','left','right','bottom',
    'width','height','min-width','min-height','max-width','max-height',
    'margin-top','margin-right','margin-bottom','margin-left',
    'padding-top','padding-right','padding-bottom','padding-left',
    'color','background-color','background-image',
    'font-size','font-weight','font-family','font-style',
    'text-align','text-decoration','line-height','letter-spacing',
    'border-top-width','border-right-width','border-bottom-width','border-left-width',
    'border-top-color','border-radius','box-shadow',
    'opacity','transform','z-index','overflow-x','overflow-y',
    'flex-direction','justify-content','align-items',
    'white-space','text-overflow','pointer-events','cursor','content'
  ];

  // ── 1. stamp interactive element indexes on the live DOM (same rule the
  //    snapshot uses) so data-a2e-eid survives into the captured markup.
  const nodes = Array.from(document.querySelectorAll(sel));
  const visible = nodes.filter((el) => {
    const r = el.getBoundingClientRect();
    const cs = window.getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    return !(r.width === 0 && r.height === 0);
  });
  const interactive = [];
  for (let i = 0; i < visible.length; i++) {
    visible[i].setAttribute('data-a2e-eid', String(i));
    const label = (visible[i].getAttribute('aria-label')
                   || visible[i].innerText || visible[i].value
                   || visible[i].getAttribute('placeholder') || '');
    interactive.push({
      index: i,
      tag: visible[i].tagName.toLowerCase(),
      name: visible[i].getAttribute('name') || '',
      href: visible[i].getAttribute('href') || '',
      text: (label || '').trim().slice(0, 120),
    });
  }

  // ── 2. every stylesheet, inline and external. Cross-origin rules throw on
  //    cssRules access — record that explicitly rather than silently empty.
  const stylesheets = [];
  let cssBytes = 0;
  for (const sheet of Array.from(document.styleSheets || [])) {
    const href = sheet.href || '(inline)';
    let text = '';
    let readable = true;
    try {
      const rules = Array.from(sheet.cssRules || []);
      text = rules.map((r) => r.cssText).join('\n');
    } catch (e) {
      readable = false;
      text = '/* unreadable stylesheet (cross-origin): ' + href + ' */';
    }
    cssBytes += text.length;
    stylesheets.push({ href, text, readable, rules: text ? text.split('\n').length : 0 });
  }

  // ── 3. flatten computed styles onto a CLONE (never mutate the live page).
  const orig = document.documentElement;
  const clone = orig.cloneNode(true);
  const oEls = Array.from(orig.querySelectorAll('*'));
  const cEls = Array.from(clone.querySelectorAll('*'));
  let stamped = 0;
  for (let i = 0; i < oEls.length && i < cEls.length; i++) {
    const cs = getComputedStyle(oEls[i]);
    let decl = '';
    for (const p of PROPS) {
      let v = '';
      try { v = cs.getPropertyValue(p); } catch (e) { v = ''; }
      if (v) decl += p + ':' + v + ';';
    }
    cEls[i].setAttribute('style', decl);
    const eid = oEls[i].getAttribute('data-a2e-eid');
    if (eid) cEls[i].setAttribute('data-a2e-eid', eid);
    if (oEls[i].id) cEls[i].setAttribute('data-a2e-id', oEls[i].id);
    stamped++;
  }

  let html = '<!DOCTYPE html>\n' + clone.outerHTML;
  let truncated = false;
  if (html.length > maxBytes) {
    html = html.slice(0, maxBytes);
    const cut = html.lastIndexOf('>');
    if (cut > 0) html = html.slice(0, cut + 1);
    truncated = true;
  }
  return {
    url: location.href,
    title: document.title,
    html,
    stylesheets,
    interactive,
    element_count: oEls.length,
    stamped,
    css_bytes: cssBytes,
    stylesheet_count: stylesheets.length,
    truncated,
  };
}
"""


class PlaywrightBrowser(BrowserBackend):
    """Headless Chromium over Playwright's sync API."""

    name = "playwright"

    def __init__(
        self,
        headless: bool = True,
        browser: str = "chromium",
        viewport: Optional[Dict[str, int]] = None,
        user_agent: str = "",
        timeout_ms: int = 15000,
        **_ignored: Any,
    ) -> None:
        self._headless = bool(headless)
        self._browser_name = browser or "chromium"
        self._viewport = viewport or {"width": 1280, "height": 900}
        self._user_agent = user_agent
        self._default_timeout = int(timeout_ms)
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        # Console capture state — initialised in __init__ (not start()) so
        # console() works before/without a launch and never AttributeErrors.
        self._console_lock = threading.Lock()
        self._console_entries: List[Dict[str, Any]] = []
        self._console_max = 500
        self._console_dropped = 0
        self._console_total = 0

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "playwright backend requires the 'playwright' package and its "
                "browsers: `pip install playwright && playwright install "
                "chromium`. Install the env image with the browser extra."
            ) from exc

        self._pw = sync_playwright().start()
        launcher = getattr(self._pw, self._browser_name)
        launch_kwargs: Dict[str, Any] = {"headless": self._headless}
        self._browser = launcher.launch(
            args=["--no-sandbox", "--disable-dev-shm-usage"],
            **launch_kwargs,
        )
        context_kwargs: Dict[str, Any] = {"viewport": self._viewport}
        if self._user_agent:
            context_kwargs["user_agent"] = self._user_agent
        self._context = self._browser.new_context(**context_kwargs)
        self._context.set_default_timeout(self._default_timeout)
        self._page = self._context.new_page()
        # ── console + page-error capture (episode-scoped, see console()) ──
        # Attached once per page; reset_session() clears the buffer rather
        # than re-attaching, so listeners are never duplicated.
        self._page.on("console", self._on_console)
        self._page.on("pageerror", self._on_page_error)
        logger.info("playwright browser started (%s headless=%s)",
                    self._browser_name, self._headless)

    def _on_console(self, msg) -> None:
        """Playwright console handler. Runs on the driver thread."""
        try:
            args = []
            for a in (getattr(msg, "args", None) or []):
                try:
                    args.append(a.json_value())
                except Exception:
                    args.append(str(a))
            with self._console_lock:
                self._console_entries.append({
                    "type": str(getattr(msg, "type", "log")),
                    "text": (msg.text if getattr(msg, "text", None)
                             else " ".join(str(a) for a in args)),
                    "args": args,
                    "ts": time.time(),
                })
                # Bound the buffer so a chatty page cannot grow memory without
                # limit (keeps the newest N; drop is recorded).
                self._console_total += 1
                if len(self._console_entries) > self._console_max:
                    dropped = len(self._console_entries) - self._console_max
                    del self._console_entries[:dropped]
                    self._console_dropped += dropped
        except Exception:  # pragma: no cover - never break a page action
            pass

    def _on_page_error(self, err) -> None:
        """Uncaught JS exception — usually the reason an action did nothing."""
        try:
            with self._console_lock:
                self._console_entries.append({
                    "type": "pageerror",
                    "text": str(err),
                    "args": [],
                    "ts": time.time(),
                })
                self._console_total += 1
                if len(self._console_entries) > self._console_max:
                    del self._console_entries[:len(self._console_entries)
                                               - self._console_max]
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        for closer in (
            lambda: self._context and self._context.close(),
            lambda: self._browser and self._browser.close(),
            lambda: self._pw and self._pw.stop(),
        ):
            try:
                closer()
            except Exception:  # pragma: no cover - best effort teardown
                logger.debug("playwright teardown step failed", exc_info=True)
        self._pw = self._browser = self._context = self._page = None

    def reset_session(self, url: str, timeout_ms: int = 15000) -> ActionResult:
        self.start()
        # A new episode starts with a clean console so captures are never
        # mixed across episodes.
        self.clear_console()
        try:
            # Clear per-episode state so episodes are independent.
            self._context.clear_cookies()
            self._page.goto(url, wait_until="domcontentloaded",
                            timeout=timeout_ms or self._default_timeout)
            self._settle()
            return ActionResult(True, "reset", detail=f"opened {self.url}",
                                navigated=True)
        except Exception as exc:
            return ActionResult(False, "reset", error=_short(exc))

    # ── console capture ────────────────────────────────────────────────────
    def clear_console(self) -> None:
        with self._console_lock:
            self._console_entries.clear()
            self._console_dropped = 0
            self._console_total = 0

    def console(self, max_entries: int = 200, clear: bool = False) -> ActionResult:
        with self._console_lock:
            entries = list(self._console_entries[-max_entries:])
            dropped = self._console_dropped
            # total is monotonic since the last clear, so a caller can compute
            # "what is new since I last looked" even when the buffer trimmed.
            total = self._console_total
            if clear:
                self._console_entries.clear()
                self._console_dropped = 0
                self._console_total = 0
        return ActionResult(
            True, "console",
            detail=f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}"
                   f"{f' ({dropped} dropped)' if dropped else ''}",
            value={"entries": entries, "count": len(entries), "total": total,
                   "dropped": dropped, "cleared": clear},
        )

    # ── state ──────────────────────────────────────────────────────────────
    @property
    def url(self) -> str:
        return self._page.url if self._page is not None else ""

    @property
    def title(self) -> str:
        try:
            return self._page.title() if self._page is not None else ""
        except Exception:
            return ""

    def snapshot(self, max_text_chars: int = 4000,
                 max_elements: int = 120) -> PageSnapshot:
        self.start()
        try:
            info = self._page.evaluate(
                _SNAPSHOT_JS, {"sel": _INTERACTIVE_SELECTOR,
                               "maxElements": max_elements}
            )
            text = self._page.evaluate(_TEXT_JS, max_text_chars)
        except Exception as exc:  # pragma: no cover - page race
            return PageSnapshot(url=self.url, title=self.title,
                                text=f"<snapshot failed: {_short(exc)}>")
        elements = [
            ElementInfo(
                index=int(e.get("index", i)),
                tag=str(e.get("tag", "")),
                role=str(e.get("role", "")),
                name=str(e.get("name", "")),
                text=str(e.get("text", "")),
                type=str(e.get("type", "")),
                value=str(e.get("value", "")),
                href=str(e.get("href", "")),
                disabled=bool(e.get("disabled", False)),
            )
            for i, e in enumerate(info.get("elements", []))
        ]
        return PageSnapshot(
            url=str(info.get("url", self.url)),
            title=str(info.get("title", self.title)),
            text=str(text.get("text", "")),
            elements=elements,
            truncated_text=bool(text.get("truncated", False)),
            truncated_elements=len(elements) >= max_elements,
        )

    # ── actions ────────────────────────────────────────────────────────────
    def navigate(self, url: str, timeout_ms: int = 15000) -> ActionResult:
        try:
            self._page.goto(url, wait_until="domcontentloaded",
                            timeout=timeout_ms or self._default_timeout)
            self._settle()
            return ActionResult(True, "navigate", detail=f"opened {self.url}",
                                navigated=True)
        except Exception as exc:
            return ActionResult(False, "navigate", error=_short(exc))

    def click(self, selector: str, timeout_ms: int = 10000) -> ActionResult:
        try:
            before = self.url
            self._page.click(selector, timeout=timeout_ms)
            self._settle()
            return ActionResult(True, "click", detail=f"clicked {selector}",
                                navigated=self.url != before)
        except Exception as exc:
            return ActionResult(False, "click", error=_short(exc))

    def fill(self, selector: str, text: str, timeout_ms: int = 10000) -> ActionResult:
        try:
            self._page.fill(selector, text, timeout=timeout_ms)
            return ActionResult(True, "fill", detail=f"filled {selector}",
                                value=text)
        except Exception as exc:
            return ActionResult(False, "fill", error=_short(exc))

    def press(self, key: str, selector: str = "") -> ActionResult:
        try:
            if selector:
                self._page.press(selector, key)
            else:
                self._page.keyboard.press(key)
            self._settle()
            return ActionResult(True, "press", detail=f"pressed {key}")
        except Exception as exc:
            return ActionResult(False, "press", error=_short(exc))

    def scroll(self, direction: str = "down", amount: int = 500) -> ActionResult:
        try:
            dy = amount if direction == "down" else -amount
            if direction in ("top", "bottom"):
                dy = 0 if direction == "top" else 10 ** 6
            self._page.mouse.wheel(0, dy)
            self._settle()
            return ActionResult(True, "scroll", detail=f"scrolled {direction}")
        except Exception as exc:
            return ActionResult(False, "scroll", error=_short(exc))

    def back(self) -> ActionResult:
        try:
            self._page.go_back()
            self._settle()
            return ActionResult(True, "back", detail="went back", navigated=True)
        except Exception as exc:
            return ActionResult(False, "back", error=_short(exc))

    def forward(self) -> ActionResult:
        try:
            self._page.go_forward()
            self._settle()
            return ActionResult(True, "forward", detail="went forward", navigated=True)
        except Exception as exc:
            return ActionResult(False, "forward", error=_short(exc))

    def reload(self) -> ActionResult:
        try:
            self._page.reload(wait_until="domcontentloaded")
            self._settle()
            return ActionResult(True, "reload", detail="reloaded", navigated=True)
        except Exception as exc:
            return ActionResult(False, "reload", error=_short(exc))

    def extract(self, selector: str) -> ActionResult:
        try:
            sel = selector or "body"
            node = self._page.query_selector(sel)
            if node is None:
                return ActionResult(False, "extract", error=f"no match: {sel}")
            text = node.inner_text()
            return ActionResult(True, "extract", detail=f"extracted {sel}",
                                value=text)
        except Exception as exc:
            return ActionResult(False, "extract", error=_short(exc))

    def evaluate(self, js: str) -> ActionResult:
        try:
            value = self._page.evaluate(js)
            return ActionResult(True, "evaluate", detail="evaluated js",
                                value=value)
        except Exception as exc:
            return ActionResult(False, "evaluate", error=_short(exc))

    def wait_for(self, selector: str, timeout_ms: int = 5000) -> ActionResult:
        try:
            self._page.wait_for_selector(selector, timeout=timeout_ms)
            return ActionResult(True, "wait_for", detail=f"found {selector}")
        except Exception as exc:
            return ActionResult(False, "wait_for", error=_short(exc))

    def screenshot(self, full_page: bool = False,
                   max_bytes: int = 1_500_000) -> ActionResult:
        try:
            raw = self._page.screenshot(full_page=full_page, type="png")
            b64 = base64.b64encode(raw).decode("ascii")
            truncated = len(raw) > max_bytes
            if truncated:
                b64 = ""
            return ActionResult(
                True, "screenshot",
                detail=f"{len(raw)} bytes{' (omitted, over cap)' if truncated else ''}",
                value={"encoding": "base64", "mime": "image/png",
                       "bytes": len(raw), "data": b64,
                       "omitted": truncated},
            )
        except Exception as exc:
            return ActionResult(False, "screenshot", error=_short(exc))

    def html(self, max_bytes: int = 500_000) -> ActionResult:
        """Style-flattened DOM capture, rich enough to diff two captures.

        Returns markup whose every element carries its COMPUTED style inline,
        plus the raw stylesheets and the interactive-element index. Two captures
        therefore differ whenever ANY resolved style, attribute, text node or
        stylesheet rule changed — not just when the markup moved.
        """
        try:
            info = self._page.evaluate(_DOM_JS, {
                "maxBytes": max_bytes,
                "sel": _INTERACTIVE_SELECTOR,
            })
        except Exception as exc:
            return ActionResult(False, "dom", error=_short(exc))
        html = str(info.get("html", ""))
        return ActionResult(
            True, "dom",
            detail=f"{len(html)} chars over "
                   f"{info.get('stylesheet_count', 0)} stylesheet(s)"
                   f"{' (truncated)' if info.get('truncated') else ''}",
            value={
                "url": str(info.get("url", self.url)),
                "title": str(info.get("title", self.title)),
                "html": html,
                "stylesheets": info.get("stylesheets") or [],
                "interactive": info.get("interactive") or [],
                "element_count": int(info.get("element_count", 0)),
                "css_bytes": int(info.get("css_bytes", 0)),
                "stylesheet_count": int(info.get("stylesheet_count", 0)),
                "truncated": bool(info.get("truncated", False)),
            },
        )

    # ── internal ───────────────────────────────────────────────────────────
    def _settle(self, timeout_ms: int = 4000) -> None:
        """Best-effort settle: wait for load state, never fail the action."""
        try:
            self._page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
        except Exception:
            pass


def _short(exc: Exception, n: int = 300) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    return msg if len(msg) <= n else msg[: n - 1] + "…"