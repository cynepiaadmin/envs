"""DOM diffing — compare two style-flattened captures from ``dom``.

Why this lives in the host: a browser environment needs "did the page change?"
as a *reward signal* and as a verification primitive, and the capture format is
this repo's contract. Keeping the diff next to the capture means a format change
breaks a test here instead of silently producing "no difference" for every pair.

Inputs are the ``value`` dict the ``dom`` action returns::

    {"html": ..., "stylesheets": [...], "interactive": [...],
     "checksum": "sha256:...", ...}

``checksum`` makes the common case free: identical checksums ⇒ identical
captures, so a full diff is skipped. Otherwise the diff walks the markup and the
stylesheet text and reports what moved.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from typing import Any, Dict, List, Tuple


def checksum(capture: Dict[str, Any]) -> str:
    """Stable sha256 over the diff-relevant payload (markup + all CSS).

    Deliberately excludes volatile fields (timing, byte counts) so the value
    only changes when something a diff would report has changed.
    """
    payload = "\n".join([
        str(capture.get("url", "")),
        str(capture.get("title", "")),
        str(capture.get("html", "")),
        "\n".join(f"{s.get('href','')}|{s.get('text','')}"
                  for s in (capture.get("stylesheets") or [])),
    ])
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _norm_html(html: str) -> List[str]:
    """Split markup into one token-ish line per element/attribute chunk.

    Raw difflib over minified/serialized HTML produces useless single-block
    diffs. Splitting on tag boundaries makes the change surface readable.
    """
    if not html:
        return []
    lines = re.split(r"(?=<)", html)
    return [ln for ln in lines if ln.strip()]


def diff_captures(before: Dict[str, Any], after: Dict[str, Any],
                  *, max_lines: int = 60) -> Dict[str, Any]:
    """Compare two captures.

    Returns::

        {"changed": bool,
         "checksum_equal": bool,
         "added_elements": [...], "removed_elements": [...],
         "stylesheet_changed": [href,...],
         "interactive_changed": bool,
         "summary": str,
         "unified": [str, ...]}

    Field-level signals are provided alongside the text diff because a caller
    usually wants one specific answer ("did the CSS change?" / "did an element
    appear?") rather than a wall of text.
    """
    cb_before = before.get("checksum") or checksum(before)
    cb_after = after.get("checksum") or checksum(after)
    if cb_before == cb_after:
        return {
            "changed": False,
            "checksum_equal": True,
            "added_elements": [],
            "removed_elements": [],
            "stylesheet_changed": [],
            "interactive_changed": False,
            "summary": "identical captures",
            "unified": [],
        }

    # ── stylesheets, keyed by href so an edit is attributable ─────────────
    sheets_before = {s.get("href", f"(#{i})"): s for i, s in
                     enumerate(before.get("stylesheets") or [])}
    sheets_after = {s.get("href", f"(#{i})"): s for i, s in
                    enumerate(after.get("stylesheets") or [])}
    stylesheet_changed = sorted(
        {href for href in set(sheets_before) | set(sheets_after)
         if sheets_before.get(href, {}).get("text")
         != sheets_after.get(href, {}).get("text")}
    )

    # ── interactive elements: identity by (tag, name|href, text) ──────────
    def _ident(entries) -> set:
        return {(e.get("tag"), e.get("name") or e.get("href"), e.get("text"))
                for e in (entries or [])}

    id_before = _ident(before.get("interactive"))
    id_after = _ident(after.get("interactive"))
    added = sorted(str(x) for x in (id_after - id_before))
    removed = sorted(str(x) for x in (id_before - id_after))

    # ── text diff over the flattened markup ───────────────────────────────
    unified = list(difflib.unified_diff(
        _norm_html(str(before.get("html", ""))),
        _norm_html(str(after.get("html", ""))),
        fromfile="before", tofile="after", lineterm="",
    ))[:max_lines]

    parts: List[str] = []
    if stylesheet_changed:
        parts.append(f"stylesheets changed: {stylesheet_changed}")
    if added:
        parts.append(f"+{len(added)} element(s)")
    if removed:
        parts.append(f"-{len(removed)} element(s)")
    if not parts:
        parts.append("markup or styling changed")

    return {
        "changed": True,
        "checksum_equal": False,
        "added_elements": added,
        "removed_elements": removed,
        "stylesheet_changed": stylesheet_changed,
        "interactive_changed": bool(added or removed),
        "summary": "; ".join(parts),
        "unified": unified,
    }


def has_changed(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    """Cheap boolean: True when anything a diff reports has changed."""
    cb_before = before.get("checksum") or checksum(before)
    cb_after = after.get("checksum") or checksum(after)
    return cb_before != cb_after


__all__ = ["checksum", "diff_captures", "has_changed"]