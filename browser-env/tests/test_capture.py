"""Screenshot / DOM capture / console / recording — the observability surface.

These are the three things an agent actually needs to debug a page it cannot
see: pixels (screenshot), structure + styling (DOM, diffable), and what the page
itself said (console). Plus the episode recording artifact.

All offline (scripted backend), all through the real A2E RPC path.
"""

from __future__ import annotations

import json
import os

import pytest

from a2e.caps.env.protocol import EnvResetRequest, EnvStepRequest

from browser_env import actions as A
from browser_env import actions as A
from browser_env.domdiff import checksum, diff_captures, has_changed

# Tasks live OUTSIDE the environment (browser_tasks/ dataset); the suite is
# the caller and supplies the full spec on reset.
from pathlib import Path

_TASKS_FILE = Path(__file__).resolve().parents[1] / "browser_tasks" / "tasks.json"
TASKS = {t["name"]: t for t in json.loads(_TASKS_FILE.read_text())["tasks"]}


def reset(client, **options):
    name = options.get("task")
    if isinstance(name, str):
        options = dict(options, task=TASKS[name])
    resp = client.rpc(EnvResetRequest(env_name="browser", options=options),
                      timeout=15)
    return resp.obs


def step(client, episode_id, action, timeout=20):
    resp = client.rpc(EnvStepRequest(episode_id=episode_id, action=action),
                      timeout=timeout)
    return resp.obs


# ── screenshot ─────────────────────────────────────────────────────────────
def test_screenshot_returns_decodable_png(host):
    obs = reset(host.client, task="add-widget-to-cart")
    obs = step(host.client, obs.episode_id,
               {"action_type": "screenshot", "payload": {}})
    shot = obs.metadata["render"]
    assert shot["mime"] == "image/png"
    assert shot["encoding"] == "base64"
    assert shot["data"], "screenshot omitted — over the byte cap?"
    import base64
    png = base64.b64decode(shot["data"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n"), "not a PNG"


def test_screenshot_served_by_the_tool_surface(host):
    from a2e.caps.tools.protocol import ToolCallRequest
    from a2e.caps.env.protocol import EnvResetRequest as ER
    host.client.rpc(ER(env_name="browser",
                       options={"task": TASKS["add-widget-to-cart"]}), timeout=15)
    resp = host.client.rpc(ToolCallRequest(tool_name="browser_screenshot",
                                           arguments={}), timeout=20)
    payload = resp.data.data
    assert payload["success"] is True, payload.get("error")
    assert payload["data"]["screenshot"]["mime"] == "image/png"


# ── DOM capture ────────────────────────────────────────────────────────────
def test_dom_capture_has_styles_and_checksum(host):
    obs = reset(host.client, task="add-widget-to-cart")
    obs = step(host.client, obs.episode_id,
               {"action_type": "dom", "payload": {}})
    dom = obs.metadata["dom"]

    # markup
    assert dom["html"].startswith("<!DOCTYPE html>")
    assert "Widget" in dom["html"]
    # COMPUTED styles are written inline (that is what makes it diffable)
    assert 'style="' in dom["html"], "no computed styles captured"
    assert "color:#0066cc" in dom["html"]
    # the raw stylesheet is captured too
    assert dom["stylesheet_count"] >= 1
    assert any("font-size" in s["text"] for s in dom["stylesheets"])
    assert dom["css_bytes"] > 0
    # element indexes survive into the markup
    assert 'data-a2e-eid="1"' in dom["html"]
    assert dom["interactive"], "interactive index missing"
    # checksum for cheap equality checks
    assert dom["checksum"].startswith("sha256:")


def test_dom_capture_is_deterministic_and_changes_on_navigation(host):
    """The whole point: same page ⇒ same checksum, different page ⇒ different."""
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    a = step(host.client, eid, {"action_type": "dom", "payload": {}}).metadata["dom"]
    b = step(host.client, eid, {"action_type": "dom", "payload": {}}).metadata["dom"]
    assert a["checksum"] == b["checksum"], "unchanged page must not change"

    step(host.client, eid, {"action_type": "navigate",
                            "payload": {"url": "https://shop.test/product/1"}})
    c = step(host.client, eid, {"action_type": "dom", "payload": {}}).metadata["dom"]
    assert c["checksum"] != a["checksum"], "navigation must change the capture"
    assert has_changed(a, c)


def test_dom_diff_reports_what_changed(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    # first capture establishes the baseline
    step(host.client, eid, {"action_type": "dom", "payload": {}})

    # an unchanged diff must say so explicitly
    same = step(host.client, eid,
                {"action_type": "dom", "payload": {"mode": "diff"}})
    d = same.metadata["dom"]["diff"]
    assert d["changed"] is False and d["checksum_equal"] is True
    assert d["summary"] == "identical captures"

    # now change the page and diff again
    step(host.client, eid, {"action_type": "click", "payload": {"index": 1}})
    diffed = step(host.client, eid,
                  {"action_type": "dom", "payload": {"mode": "diff"}})
    d = diffed.metadata["dom"]["diff"]
    assert d["changed"] is True
    assert d["checksum_equal"] is False
    # the summary names what moved (elements appearing/vanishing, CSS edits)
    assert d["summary"] and d["summary"] != "identical captures"
    assert any(k in d["summary"] for k in ("element", "stylesheets", "changed"))
    assert d["unified"], "no unified diff produced"
    assert d["interactive_changed"] or d["stylesheet_changed"]


def test_dom_diff_without_baseline_fails_clearly(host):
    obs = reset(host.client, task="add-widget-to-cart")
    res = step(host.client, obs.episode_id,
               {"action_type": "dom", "payload": {"mode": "diff"}})
    # surfaced as an action error, episode stays alive
    assert res.metadata.get("action_ok") is False
    assert "mode=capture" in res.state.model_dump()["last_error"]
    assert res.done is False


def test_dom_diff_resets_between_episodes(host):
    """A new episode must not diff against the previous episode's page."""
    obs1 = reset(host.client, task="add-widget-to-cart")
    step(host.client, obs1.episode_id,
         {"action_type": "dom", "payload": {}})          # baseline in ep 1
    reset(host.client, task="read-widget-price")          # new episode

    res = step(host.client, "", {"action_type": "dom",
                                 "payload": {"mode": "diff"}})
    # The episode id above is unused: the host binds by active episode. What
    # matters is that a fresh episode starts with NO baseline, so its first
    # diff reports an error rather than comparing against the old page.
    assert res.metadata.get("action_ok") is False
    assert "no previous capture" in res.state.model_dump()["last_error"]


def test_dom_unknown_mode_is_rejected(host):
    obs = reset(host.client, task="add-widget-to-cart")
    res = step(host.client, obs.episode_id,
               {"action_type": "dom", "payload": {"mode": "sideways"}})
    assert res.metadata.get("action_ok") is False
    assert "capture|diff" in res.state.model_dump()["last_error"]


# ── console output ─────────────────────────────────────────────────────────
def test_console_captured_on_reset(host):
    """The episode's load line is readable right after reset."""
    obs = reset(host.client, task="add-widget-to-cart")
    res = step(host.client, obs.episode_id,
               {"action_type": "console", "payload": {}})
    value = res.metadata.get("console")
    assert value, f"console payload missing: {sorted(res.metadata)}"
    assert value["count"] >= 1
    assert any("page loaded" in e["text"] for e in value["entries"])


def test_console_surfaces_page_warnings_and_is_episode_scoped(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    # the cart page authors a warning; it appears after navigating there
    step(host.client, eid, {"action_type": "navigate",
                            "payload": {"url": "https://shop.test/cart"}})
    res = step(host.client, eid, {"action_type": "console", "payload": {}})
    entries = res.metadata["console"]["entries"]
    assert any(e["type"] == "warn" and "localStorage" in e["text"]
               for e in entries), entries
    assert any("page loaded" in e["text"] for e in entries)


def test_console_delta_appears_in_step_metadata(host):
    """New console output must ride along with the step that caused it."""
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    step(host.client, eid, {"action_type": "click", "payload": {"index": 1}})
    obs = step(host.client, eid, {"action_type": "click", "payload": {"index": 0}})
    console = obs.metadata.get("console")
    assert console, "expected a console delta after adding to cart"
    assert any("cart: added Widget" in e["text"] for e in console["new"]), \
        console["new"]
    assert console["total"] >= len(console["new"])


def test_console_read_is_idempotent_by_default(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    first = step(host.client, eid, {"action_type": "console", "payload": {}})
    second = step(host.client, eid, {"action_type": "console", "payload": {}})
    assert first.metadata["console"]["count"] == \
        second.metadata["console"]["count"], "reading twice must be stable"


def test_console_clear_empties_the_buffer(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    step(host.client, eid, {"action_type": "console", "payload": {"clear": True}})
    after = step(host.client, eid, {"action_type": "console", "payload": {}})
    assert after.metadata["console"]["count"] == 0, after.metadata["console"]
    # a clear must not break the delta cursor: new output still shows up
    step(host.client, eid, {"action_type": "navigate",
                            "payload": {"url": "https://shop.test/cart"}})
    later = step(host.client, eid, {"action_type": "console", "payload": {}})
    assert later.metadata["console"]["count"] > 0


def test_console_new_episode_starts_clean(host):
    """Console output must never leak across episodes."""
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    # make ep 1 noisy: the cart page authors a warning
    step(host.client, eid, {"action_type": "navigate",
                            "payload": {"url": "https://shop.test/cart"}})
    noisy = step(host.client, eid, {"action_type": "console", "payload": {}})
    assert "localStorage" in " ".join(
        e["text"] for e in noisy.metadata["console"]["entries"])

    # a new episode starts clean: only its own load line, no cart warning
    reset(host.client, task="read-widget-price")
    res = step(host.client, "", {"action_type": "console", "payload": {}})
    entries = res.metadata["console"]["entries"]
    texts = " ".join(e["text"] for e in entries)
    assert "localStorage" not in texts, entries
    assert "page loaded" in texts


# ── recording ──────────────────────────────────────────────────────────────
def test_record_start_step_stop_writes_jsonl(host, tmp_path_factory):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id

    start = step(host.client, eid, {"action_type": "record",
                                    "payload": {"mode": "start"}})
    rec = start.metadata["record"]
    assert rec["recording"] is True
    assert rec["path"].endswith(".jsonl")

    step(host.client, eid, {"action_type": "click", "payload": {"index": 1}})
    step(host.client, eid, {"action_type": "click", "payload": {"index": 0}})
    step(host.client, eid, {"action_type": "navigate",
                            "payload": {"url": "https://shop.test/cart"}})
    # finish terminates the episode AND auto-stops the recorder, reporting the
    # artifact in the terminal observation (you cannot step a finished episode).
    done = step(host.client, eid, {"action_type": "finish", "payload": {}})
    rec = done.metadata["record"]
    assert rec["recording"] is False, rec
    assert rec["entries"] >= 4, rec

    # the file exists and parses line by line
    path = rec["path"]
    assert os.path.exists(path), path
    with open(path) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    assert len(rows) >= 3, [r["action"] for r in rows]
    actions = [r["action"] for r in rows]
    assert "click" in actions and "finish" in actions
    # terminal row carries the reward/done flags
    assert rows[-1]["done"] is True
    assert rows[-1]["reward"] == 1.0
    # every row is a full transition record
    for r in rows:
        assert set(r) >= {"step", "action", "args", "url", "reward", "done", "ts"}


def test_record_status_before_start_says_not_recording(host):
    obs = reset(host.client, task="add-widget-to-cart")
    res = step(host.client, obs.episode_id, {"action_type": "record",
                                             "payload": {"mode": "status"}})
    assert res.metadata["record"]["recording"] is False


def test_record_with_screenshots_embeds_frames(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    step(host.client, eid, {"action_type": "record",
                            "payload": {"mode": "start",
                                        "include_screenshots": True}})
    step(host.client, eid, {"action_type": "click", "payload": {"index": 1}})
    stop = step(host.client, eid, {"action_type": "record",
                                   "payload": {"mode": "stop"}})
    rec = stop.metadata["record"]
    with open(rec["path"]) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    framed = [r for r in rows if r.get("screenshot")]
    assert framed, "no screenshot frames embedded despite include_screenshots"
    assert framed[0]["screenshot"]["mime"] == "image/png"
    assert framed[0]["screenshot"]["data"]


def test_record_rejects_unknown_mode(host):
    obs = reset(host.client, task="add-widget-to-cart")
    res = step(host.client, obs.episode_id, {"action_type": "record",
                                             "payload": {"mode": "rewind"}})
    assert res.metadata.get("action_ok") is False
    assert "start|stop|status" in res.state.model_dump()["last_error"]


# ── tool surface parity for all four ───────────────────────────────────────
def test_tool_surface_exposes_all_capture_tools(host):
    from browser_env.browser_tool import BrowserToolPlugin
    names = {t.name for t in host.tool_plugin._list_tools()}
    assert {"browser_dom", "browser_console", "browser_record",
            "browser_screenshot"} <= names


def test_tool_dom_and_console_route_through_the_episode(host):
    """Tool calls must consume episode steps — the two surfaces agree."""
    from a2e.caps.env.protocol import EnvResetRequest as ER
    from a2e.caps.tools.protocol import ToolCallRequest
    host.client.rpc(ER(env_name="browser",
                       options={"task": TASKS["add-widget-to-cart"]}), timeout=15)

    r = host.client.rpc(ToolCallRequest(tool_name="browser_dom", arguments={}),
                        timeout=20)
    payload = r.data.data
    assert payload["success"] is True, payload.get("error")
    assert payload["data"]["step_num"] == 1, "tool call did not consume a step"
    assert payload["data"]["dom"]["checksum"].startswith("sha256:")

    r = host.client.rpc(ToolCallRequest(tool_name="browser_console",
                                        arguments={}), timeout=20)
    payload = r.data.data
    assert payload["success"] is True, payload.get("error")
    assert payload["data"]["step_num"] == 2


# ── unit tests for the diff helper ─────────────────────────────────────────
def test_diff_helper_reports_stylesheet_only_change():
    before = {"url": "https://x/", "title": "T",
              "html": "<html><body><a style='color:red'>x</a></body></html>",
              "stylesheets": [{"href": "(inline)", "text": "a{color:red}"}],
              "interactive": [{"tag": "a", "name": "", "href": "", "text": "x"}]}
    after = dict(before)
    after["stylesheets"] = [{"href": "(inline)", "text": "a{color:blue}"}]
    after["html"] = before["html"].replace("color:red", "color:blue")
    d = diff_captures(before, after)
    assert d["changed"] is True
    assert d["stylesheet_changed"] == ["(inline)"]
    assert d["interactive_changed"] is False
    assert "stylesheets changed" in d["summary"]


def test_diff_helper_reports_added_and_removed_elements():
    before = {"url": "u", "title": "t", "html": "<a>x</a>",
              "stylesheets": [], "interactive": []}
    after = {"url": "u", "title": "t", "html": "<a>x</a><button>y</button>",
             "stylesheets": [],
             "interactive": [{"tag": "button", "name": "b", "href": "",
                              "text": "y"}]}
    d = diff_captures(before, after)
    assert d["changed"] is True
    assert len(d["added_elements"]) == 1
    assert d["interactive_changed"] is True


def test_checksum_ignores_volatile_fields():
    a = {"url": "u", "title": "t", "html": "<p>x</p>",
         "stylesheets": [], "elapsed": 123}
    b = {"url": "u", "title": "t", "html": "<p>x</p>",
         "stylesheets": [], "elapsed": 456}
    assert checksum(a) == checksum(b)