"""Task SPEC validation + deterministic scoring for the browser environment.

A task is a goal plus explicit success criteria. The spec is supplied by the
CALLER on ``env/reset`` (tasks live outside the environment — the host keeps no
catalogue and never loads a task file). Nothing is inferred:

    {
      "name": "add-widget-to-cart",
      "start_url": "https://shop.test/",
      "goal": "Add the Widget to the cart and finish on the cart page.",
      "max_steps": 15,
      "success": {
        "url_contains": ["/cart"],
        "text_contains": ["widget"],
        "required_actions": ["click"],
        "max_steps_within": 12
      },
      "keywords": ["cart", "widget"]
    }

Every key under ``success`` is checked against the episode record; a criterion
that is not present is simply not enforced (no silent default flipped to True).
``score_episode`` returns ONE number in [0, 1] — the fraction of declared
criteria met — plus the per-criterion verdicts so a failure is explainable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

#: Criteria keys the scorer understands. Unknown keys are a hard error so a
#: typo in a caller-supplied spec fails loudly instead of silently never
#: being checked.
KNOWN_CRITERIA = {
    "url_contains",
    "url_equals",
    "title_contains",
    "text_contains",
    "text_contains_any",
    "required_actions",
    "forbidden_actions",
    "max_steps_within",
    "min_steps",
    "clicked_text_contains",
    "filled",
    "finished",
}


class TaskError(ValueError):
    """Raised for a malformed task definition."""


@dataclass
class BrowserTask:
    name: str
    start_url: str
    goal: str
    max_steps: int = 15
    success: Dict[str, Any] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        for key in self.success:
            if key not in KNOWN_CRITERIA:
                raise TaskError(
                    f"task {self.name!r}: unknown success criterion {key!r}; "
                    f"known criteria: {sorted(KNOWN_CRITERIA)}"
                )
        if not self.start_url:
            raise TaskError(f"task {self.name!r}: start_url is required")
        if not self.goal:
            raise TaskError(f"task {self.name!r}: goal is required")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BrowserTask":
        if not isinstance(raw, dict):
            raise TaskError(f"task must be an object, got {type(raw).__name__}")
        missing = [k for k in ("name", "start_url", "goal") if k not in raw]
        if missing:
            raise TaskError(f"task missing required field(s): {missing}")
        return cls(
            name=raw["name"],
            start_url=raw["start_url"],
            goal=raw["goal"],
            max_steps=int(raw.get("max_steps", 15)),
            success=dict(raw.get("success", {}) or {}),
            keywords=list(raw.get("keywords", []) or []),
            description=raw.get("description", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_url": self.start_url,
            "goal": self.goal,
            "max_steps": self.max_steps,
            "success": self.success,
            "keywords": self.keywords,
            "description": self.description,
        }


def score_episode(task: BrowserTask, record: Dict[str, Any]) -> Dict[str, Any]:
    """Score one finished episode against ``task.success``.

    ``record`` is the episode summary produced by the env plugin:
        {url, title, text, history, actions, extracted, success, done,
         step_num, max_steps, finished}

    Returns ``{"score": float, "passed": bool, "criteria": {...}}`` where
    ``score`` is the fraction of declared criteria met (1.0 when none declared —
    an explicit "no criteria" task is scored on completion alone).
    """
    crit = task.success or {}
    if not crit:
        # No declared criteria: success is simply "the agent finished".
        ok = bool(record.get("finished") or record.get("done"))
        return {"score": 1.0 if ok else 0.0,
                "passed": ok,
                "criteria": {"finished": ok}}

    text = (record.get("text") or "").lower()
    url = (record.get("url") or "").lower()
    title = (record.get("title") or "").lower()
    actions = [str(a).lower() for a in record.get("actions", [])]
    clicked = [str(c).lower() for c in record.get("clicked_text", [])]
    filled = record.get("filled", {}) or {}
    step_num = int(record.get("step_num", 0))

    verdicts: Dict[str, bool] = {}

    if "url_contains" in crit:
        verdicts["url_contains"] = all(
            str(s).lower() in url for s in crit["url_contains"])
    if "url_equals" in crit:
        verdicts["url_equals"] = url.rstrip("/") == str(crit["url_equals"]).lower().rstrip("/")
    if "title_contains" in crit:
        verdicts["title_contains"] = all(
            str(s).lower() in title for s in crit["title_contains"])
    if "text_contains" in crit:
        verdicts["text_contains"] = all(
            str(s).lower() in text for s in crit["text_contains"])
    if "text_contains_any" in crit:
        verdicts["text_contains_any"] = any(
            str(s).lower() in text for s in crit["text_contains_any"])
    if "required_actions" in crit:
        verdicts["required_actions"] = all(
            a in actions for a in [str(x).lower() for x in crit["required_actions"]])
    if "forbidden_actions" in crit:
        forbidden = [str(x).lower() for x in crit["forbidden_actions"]]
        verdicts["forbidden_actions"] = not any(a in actions for a in forbidden)
    if "max_steps_within" in crit:
        verdicts["max_steps_within"] = step_num <= int(crit["max_steps_within"])
    if "min_steps" in crit:
        verdicts["min_steps"] = step_num >= int(crit["min_steps"])
    if "clicked_text_contains" in crit:
        verdicts["clicked_text_contains"] = all(
            any(str(s).lower() in c for c in clicked)
            for s in crit["clicked_text_contains"])
    if "filled" in crit:
        verdicts["filled"] = all(
            str(name) in filled and str(val).lower() in str(filled[str(name)]).lower()
            for name, val in dict(crit["filled"]).items())
    if "finished" in crit:
        verdicts["finished"] = bool(record.get("finished") or record.get("done"))

    met = sum(1 for v in verdicts.values() if v)
    total = len(verdicts) or 1
    score = met / total
    return {"score": score, "passed": met == total, "criteria": verdicts}


__all__ = [
    "BrowserTask",
    "TaskError",
    "KNOWN_CRITERIA",
    "score_episode",
]