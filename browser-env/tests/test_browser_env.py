"""Env-plugin and browser-tool tests — offline, scripted backend.

These drive the REAL A2E protocol path (DirectTransport pair), the real SDK
EnvPlugin/ToolPlugin wiring, and the real scoring code. Only the browser
driver is swapped for the deterministic scripted model, so the suite needs no
network and no Chromium.
"""

from __future__ import annotations

import pytest

from a2e.caps.env.protocol import (
    EnvResetRequest,
    EnvResetResponse,
    EnvStepRequest,
    EnvStepResponse,
    EnvObserveRequest,
    EnvObserveResponse,
    EnvCloseRequest,
    EnvExpListRequest,
    EnvExpListResponse,
)
from a2e.caps.tools.protocol import ToolCallRequest, ToolCallResponse

from browser_env import actions as A

# Tasks live OUTSIDE the environment as a separate dataset (browser_tasks/,
# a sibling of the browser_env package). The test suite plays the caller: it
# reads the dataset and supplies the FULL spec on reset — exactly what a real
# agent-side task runner does.
import json
from pathlib import Path

_TASKS_FILE = Path(__file__).resolve().parents[1] / "browser_tasks" / "tasks.json"
TASKS = {t["name"]: t for t in json.loads(_TASKS_FILE.read_text())["tasks"]}


# ── helpers ────────────────────────────────────────────────────────────────
def reset(client, **options):
    # Translate a task NAME into the full spec object — the host resolves
    # nothing; the caller supplies the whole task.
    name = options.get("task")
    if isinstance(name, str):
        options = dict(options, task=TASKS[name])
    resp = client.rpc(EnvResetRequest(env_name="browser", options=options),
                      timeout=15)
    assert isinstance(resp, EnvResetResponse), f"unexpected: {type(resp)}"
    return resp.obs


def step(client, episode_id, action, timeout=15):
    resp = client.rpc(EnvStepRequest(episode_id=episode_id, action=action),
                      timeout=timeout)
    assert isinstance(resp, EnvStepResponse), f"unexpected: {type(resp)}"
    return resp.obs


def call_tool(client, name, args, timeout=15):
    """Call a tool and return the tool's own payload dict.

    The wire nests twice: ToolCallResponse.data is a ToolResult, and the
    tool's `_execute_tool` return value is that ToolResult's `.data`. So the
    payload (`{"success": bool, "data": ...}`) is `resp.data.data`.
    """
    resp = client.rpc(ToolCallRequest(tool_name=name, arguments=args), timeout=timeout)
    assert isinstance(resp, ToolCallResponse), f"unexpected: {type(resp)}"
    payload = resp.data.data
    assert isinstance(payload, dict), f"unexpected payload: {payload!r}"
    return payload


# ── reset ──────────────────────────────────────────────────────────────────
def test_reset_with_external_task_spec_returns_task_state(host):
    obs = reset(host.client, task="add-widget-to-cart")
    state = obs.state.model_dump()
    assert state["task"] == "add-widget-to-cart"
    assert state["url"] == "https://shop.test/"
    assert state["max_steps"] == 12          # from the supplied spec, not default
    assert state["step_num"] == 0
    assert state["done"] is False
    # reset outcome travels in the state (EnvObservation.metadata is reserved
    # for per-step info; the SDK's reset() does not carry it through)
    assert state["reset_ok"] is True
    # the page snapshot travels in the reset STATE (the SDK's reset() builds
    # its observation without metadata, so text/elements live in the state)
    assert "Widget" in state["page_text"]
    assert len(state["elements"]) >= 3


def test_reset_requires_task_or_url(host):
    """No task and no start_url must fail loudly, not default to about:blank.

    The plugin raises inside on_reset; the SDK turns that into an A2EError and
    the client surfaces it as A2EClientError. That is the correct behaviour —
    a bad reset must be an error, never a silently-empty page.
    """
    from a2e.core.client.client import A2EClientError
    with pytest.raises(A2EClientError) as ei:
        host.client.rpc(EnvResetRequest(env_name="browser", options={}),
                        timeout=15)
    assert "start_url" in str(ei.value)


def test_reset_rejects_bare_task_name(host):
    """The host keeps no catalogue: a NAME cannot be resolved. The caller must
    pass the full spec — the error says so instead of guessing."""
    from a2e.core.client.client import A2EClientError
    with pytest.raises(A2EClientError) as ei:
        host.client.rpc(
            EnvResetRequest(env_name="browser", options={"task": "nope"}),
            timeout=15)
    assert "FULL task spec" in str(ei.value)
    assert "outside" in str(ei.value)


def test_reset_adhoc_url(host):
    obs = reset(host.client, start_url="https://shop.test/cart",
                goal="look at the cart")
    assert obs.state.model_dump()["url"] == "https://shop.test/cart"


# ── step: the happy path, end to end ───────────────────────────────────────
def test_add_widget_to_cart_episode_scores_one(host):
    """Navigate to the Widget, add it, finish on the cart page -> score 1.0."""
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id

    # click the Widget link (index 1 on the home page)
    obs = step(host.client, eid, {"action_type": "click", "payload": {"index": 1}})
    assert obs.state.model_dump()["url"] == "https://shop.test/product/1"

    # click "Add to cart" (index 0 on the product page)
    obs = step(host.client, eid, {"action_type": "click", "payload": {"index": 0}})

    # go to the cart
    obs = step(host.client, eid, {"action_type": "navigate",
                                  "payload": {"url": "https://shop.test/cart"}})
    assert "widget" in obs.metadata["render_text"].lower()

    # finish
    obs = step(host.client, eid, {"action_type": "finish",
                                  "payload": {"answer": "added the widget"}})
    assert obs.done is True
    assert obs.reward == 1.0, obs.metadata
    assert obs.metadata["passed"] is True
    assert all(obs.metadata["criteria"].values()), obs.metadata["criteria"]


def test_search_episode_fill_press_scores_one(host):
    obs = reset(host.client, task="search-for-gadget")
    eid = obs.episode_id
    # fill the search box (index 0)
    obs = step(host.client, eid, {"action_type": "fill",
                                  "payload": {"index": 0, "text": "gadget"}})
    assert obs.state.model_dump()["url"] == "https://shop.test/"  # no nav yet
    # press Enter
    obs = step(host.client, eid, {"action_type": "press",
                                  "payload": {"key": "Enter"}})
    assert obs.state.model_dump()["url"] == "https://shop.test/product/2"
    obs = step(host.client, eid, {"action_type": "finish", "payload": {}})
    assert obs.done and obs.reward == 1.0, obs.metadata
    assert obs.metadata["criteria"]["filled"] is True


def test_max_steps_terminates_with_partial_score(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    # 12 steps of scrolling never satisfies the criteria
    last = None
    for _ in range(12):
        last = step(host.client, eid, {"action_type": "scroll",
                                       "payload": {"direction": "down"}})
    assert last.done is True
    assert last.metadata["terminal_reason"] == "max_steps"
    assert 0.0 <= last.reward < 1.0
    assert last.metadata["passed"] is False


# ── step: error handling must not kill the episode ─────────────────────────
def test_malformed_action_is_recorded_not_fatal(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    obs = step(host.client, eid, {"action_type": "click", "payload": {}})
    assert obs.done is False
    assert "action_error" in obs.metadata
    # the episode is still alive and can continue
    obs = step(host.client, eid, {"action_type": "back", "payload": {}})
    assert obs.done is False


def test_unknown_action_reports_valid_set(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    obs = step(host.client, eid, {"action_type": "teleport", "payload": {}})
    assert "action_error" in obs.metadata
    assert "scroll" in obs.metadata["action_error"]
    assert obs.done is False


def test_fill_on_non_input_is_an_error_tag(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    # index 1 is a link, not a field
    obs = step(host.client, eid, {"action_type": "fill",
                                  "payload": {"index": 1, "text": "x"}})
    assert obs.metadata.get("action_ok") is False
    assert obs.state.model_dump()["last_error"]


# ── observe / close / spaces ───────────────────────────────────────────────
def test_observe_returns_state_without_acting(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    seen = host.client.rpc(EnvObserveRequest(episode_id=eid), timeout=15)
    assert isinstance(seen, EnvObserveResponse)
    assert seen.obs.state.model_dump()["step_num"] == 0
    assert seen.obs.state.model_dump()["url"] == "https://shop.test/"


def test_spaces_advertises_the_full_action_set(host):
    spaces = host.env_plugin.spaces()
    for name in A.ACTION_NAMES:
        assert name in spaces["action_space"], f"{name} missing from action_space"


def test_close_ends_episode_but_keeps_browser_alive(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    resp = host.client.rpc(EnvCloseRequest(episode_id=eid), timeout=15)
    assert resp.closed is True
    # browser is process-scoped: still usable, and a new reset works
    again = reset(host.client, task="add-widget-to-cart")
    assert again.episode_id != eid


# ── transition log ─────────────────────────────────────────────────────────
def test_transitions_are_recorded(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    step(host.client, eid, {"action_type": "click", "payload": {"index": 1}})
    step(host.client, eid, {"action_type": "finish", "payload": {}})

    resp = host.client.rpc(EnvExpListRequest(episode_id=eid), timeout=15)
    assert isinstance(resp, EnvExpListResponse)
    assert len(resp.episodes) >= 2
    rows = host.env_plugin.list_experiences(eid)
    assert rows[0]["state"]["task"] == "add-widget-to-cart" or \
        rows[0]["state"].get("task") in (None, "")
    assert any(r["done"] for r in rows)


# ── scoring protocol messages ──────────────────────────────────────────────
def test_criteria_originate_from_outside(host):
    """A task spec the host has never seen binds and scores — criteria are
    caller-supplied data, not host configuration. Scoring rides the SDK tool
    surface (tool/call), no custom wire type."""
    custom = {"name": "reach-product-one",
              "start_url": "https://shop.test/",
              "goal": "Open the Widget product page.",
              "max_steps": 6,
              "success": {"url_contains": ["/product/1"], "finished": True}}
    obs = reset(host.client, task=custom)
    assert obs.state.model_dump()["task"] == "reach-product-one"
    eid = obs.episode_id
    step(host.client, eid, {"action_type": "click", "payload": {"index": 1}})
    step(host.client, eid, {"action_type": "finish", "payload": {}})
    res = call_tool(host.client, "browser_score", {})
    assert res["success"] is True, res.get("error")
    assert res["data"]["score"] == 1.0 and res["data"]["passed"] is True


def test_score_tool_ignores_agent_self_report(host):
    """The host's own record must win over caller-supplied hints."""
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    # Agent claims it is already on the cart page; it is not.
    res = call_tool(host.client, "browser_score",
                    {"record": {"url": "https://shop.test/cart",
                                "text": "widget", "finished": True}})
    assert res["success"] is True, res.get("error")
    assert res["data"]["passed"] is False, "self-report leaked into scoring"
    assert res["data"]["criteria"]["url_contains"] is False


def test_score_tool_after_real_finish(host):
    obs = reset(host.client, task="add-widget-to-cart")
    eid = obs.episode_id
    step(host.client, eid, {"action_type": "click", "payload": {"index": 1}})
    step(host.client, eid, {"action_type": "click", "payload": {"index": 0}})
    step(host.client, eid, {"action_type": "navigate",
                            "payload": {"url": "https://shop.test/cart"}})
    done = step(host.client, eid, {"action_type": "finish", "payload": {}})
    # the terminal observation itself carries the score (SDK reward path)
    assert done.metadata["score"] == 1.0 and done.metadata["passed"] is True
    res = call_tool(host.client, "browser_score", {})
    assert res["data"]["score"] == 1.0 and res["data"]["passed"] is True


# ── the tool surface ───────────────────────────────────────────────────────
def test_browser_tools_are_listed(host):
    names = {t.name for t in host.tool_plugin._list_tools()}
    assert {"browser_navigate", "browser_click", "browser_fill",
            "browser_extract", "browser_text",
            "browser_score"} <= names
    # The task catalogue belongs to the caller (browser_tasks/ dataset) —
    # the host advertises no catalogue tool and defines no custom wire type.
    assert "browser_tasks" not in names


def test_tool_navigate_then_text_roundtrip(host):
    res = call_tool(host.client, "browser_navigate",
                    {"url": "https://shop.test/product/1"})
    assert res["success"] is True, res.get("error")
    assert res["data"]["url"] == "https://shop.test/product/1"

    res = call_tool(host.client, "browser_text", {})
    assert res["success"] is True
    assert "Widget" in res["data"]["title"] or "Widget" in res["data"]["text"]


def test_tool_and_env_share_one_browser(host):
    """A tool call after env/reset must act on the SAME page."""
    obs = reset(host.client, task="add-widget-to-cart")
    res = call_tool(host.client, "browser_click", {"index": 1})
    assert res["success"] is True, res.get("error")
    # the env plugin now observes the navigated page
    seen = host.client.rpc(EnvObserveRequest(episode_id=obs.episode_id), timeout=15)
    assert seen.obs.state.model_dump()["url"] == "https://shop.test/product/1"


def test_tool_failure_returns_error_not_exception(host):
    res = call_tool(host.client, "browser_click", {"selector": "#missing"})
    assert res["success"] is False
    assert res["error"]


def test_tool_without_episode_and_no_default_refuses(host):
    """A tool call with no episode and no DEFAULT_URL must error, not
    silently drive about:blank."""
    res = call_tool(host.client, "browser_text", {})
    assert res["success"] is False
    assert "DEFAULT_URL" in res["error"] or "no browser episode" in res["error"]


def test_tool_with_default_url_self_bootstraps(make_host):
    h = make_host(default_url="https://shop.test/")
    res = call_tool(h.client, "browser_text", {})
    assert res["success"] is True, res.get("error")
    assert res["data"]["url"] == "https://shop.test/"


# ── capability negotiation ─────────────────────────────────────────────────
def test_handshake_negotiates_env_and_tools(host):
    caps = {c.capability.value if hasattr(c.capability, "value")
            else str(c.capability)
            for c in host.client.capabilities()}
    assert "env" in caps
    assert "tools" in caps