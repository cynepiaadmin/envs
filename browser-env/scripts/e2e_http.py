"""End-to-end over REAL HTTP: live uvicorn host + real A2EClient.

This is not an in-process harness — it talks to a running `python -m
browser_env.app` over /send + /stream, so it exercises the actual wire, the
session handshake, and SSE streaming. Run it against a live host:

    python -m browser_env.app --config browser_env/host_config.local.yaml &
    python scripts/e2e_http.py --base-url http://localhost:8791
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from a2e.core.client.client import A2EClient
from a2e.core.transports.http import HTTPTransport
from a2e.caps.env.protocol import (
    ENV_TYPE_MAP,
    EnvResetRequest,
    EnvResetResponse,
    EnvStepRequest,
    EnvStepResponse,
    EnvObserveRequest,
    EnvObserveResponse,
)
from a2e.caps.tools.protocol import TOOL_TYPE_MAP, ToolCallRequest, ToolCallResponse


def load_tasks() -> dict:
    """The task dataset is OUTSIDE the environment (browser_tasks/, a sibling
    of the browser_env package): this client script supplies the full spec."""
    path = Path(__file__).resolve().parents[1] / "browser_tasks" / "tasks.json"
    raw = json.loads(path.read_text())
    return {t["name"]: t for t in raw["tasks"]}


TASKS = load_tasks()

def build_client(base_url: str) -> A2EClient:
    logger = logging.getLogger("e2e")
    logger.setLevel(logging.INFO)
    transport = HTTPTransport(base_url, logger, session_id=None)
    client = A2EClient(transport=transport, logger=logger,
                       agent_id="browser-agent-http",
                       agent_caps=["env", "tools"])
    # A2EClient seeds only A2E_BASE_TYPE_MAP. Register the SDK namespaces we
    # consume (env, tools) or responses decode to a bare message and the RPC
    # hangs. No custom wire types: score rides the tool surface.
    client.update_msg_types(ENV_TYPE_MAP)
    client.update_msg_types(TOOL_TYPE_MAP)
    client.connect()
    return client


def run_add_widget_to_cart(client) -> dict:
    resp = client.rpc(EnvResetRequest(env_name="browser",
                                      options={"task": TASKS["add-widget-to-cart"]}),
                      timeout=30)
    assert isinstance(resp, EnvResetResponse), type(resp)
    eid = resp.obs.episode_id
    print(f"reset -> episode={eid} url={resp.obs.state.model_dump()['url']}")

    def step(action: dict) -> EnvStepResponse:
        r = client.rpc(EnvStepRequest(episode_id=eid, action=action), timeout=30)
        assert isinstance(r, EnvStepResponse), type(r)
        state = r.obs.state.model_dump()
        print(f"  step {r.obs.step_num}: {action.get('action_type')} -> "
              f"{state['url']} reward={r.obs.reward} done={r.obs.done}"
              + (f" err={state['last_error']}" if state['last_error'] else ""))
        return r

    step({"action_type": "click", "payload": {"index": 1}})       # open Widget
    step({"action_type": "click", "payload": {"index": 0}})       # add to cart
    step({"action_type": "navigate",
          "payload": {"url": "https://shop.test/cart"}})          # go to cart
    final = step({"action_type": "finish", "payload": {"answer": "Widget"}})

    sc = client.rpc(ToolCallRequest(tool_name="browser_score", arguments={}),
                    timeout=30)
    assert isinstance(sc, ToolCallResponse), type(sc)
    score = sc.data.data
    assert score["success"], score.get("error")

    observed = client.rpc(EnvObserveRequest(episode_id=eid), timeout=30)
    assert isinstance(observed, EnvObserveResponse), type(observed)

    return {
        "episode_id": eid,
        "terminal_reward": final.obs.reward,
        "done": final.obs.done,
        "criteria": final.obs.metadata["criteria"],
        "tool_score": score["data"]["score"],
        "tool_passed": score["data"]["passed"],
        "observe_url": observed.obs.state.model_dump()["url"],
        "passed": final.obs.done and final.obs.reward == 1.0
                  and score["data"]["passed"],
    }


def run_capture_surfaces(client) -> dict:
    """DOM + console + screenshot + record, over the real HTTP wire."""
    from a2e.caps.env.protocol import EnvStepResponse

    resp = client.rpc(EnvResetRequest(env_name="browser",
                                      options={"task": TASKS["add-widget-to-cart"]}),
                      timeout=30)
    eid = resp.obs.episode_id

    def act(action, timeout=30):
        r = client.rpc(EnvStepRequest(episode_id=eid, action=action),
                       timeout=timeout)
        assert isinstance(r, EnvStepResponse), type(r)
        return r.obs

    # screenshot
    shot = act({"action_type": "screenshot", "payload": {}})
    png = shot.metadata["render"]
    assert png["data"], "screenshot omitted"
    assert png["mime"] == "image/png"

    # DOM capture (style-flattened + checksummed)
    dom1 = act({"action_type": "dom", "payload": {}}).metadata["dom"]
    assert dom1["checksum"].startswith("sha256:")
    assert 'style="' in dom1["html"]
    assert dom1["stylesheet_count"] >= 1

    # record on
    rec = act({"action_type": "record", "payload": {"mode": "start"}})
    assert rec.metadata["record"]["recording"] is True

    # console read
    con = act({"action_type": "console", "payload": {}}).metadata["console"]
    assert con["count"] >= 1

    # act, then diff the DOM against the earlier capture
    act({"action_type": "click", "payload": {"index": 1}})
    dom2 = act({"action_type": "dom", "payload": {"mode": "diff"}}).metadata["dom"]
    assert dom2["diff"]["changed"] is True, dom2["diff"]
    assert dom2["diff"]["unified"], "no unified diff over HTTP"

    # complete the task so the terminal reward is a full score
    act({"action_type": "click", "payload": {"index": 0}})   # add to cart
    act({"action_type": "navigate",
         "payload": {"url": "https://shop.test/cart"}})
    # finish -> the artifact is finalized and reported
    done = act({"action_type": "finish", "payload": {"answer": "Widget"}})
    final_rec = done.metadata["record"]
    assert final_rec["recording"] is False
    import json as _json
    with open(final_rec["path"]) as fh:
        rows = [_json.loads(l) for l in fh if l.strip()]
    assert len(rows) >= 2

    return {
        "screenshot_bytes": png["bytes"],
        "dom_checksum": dom1["checksum"][:22] + "...",
        "dom_stylesheets": dom1["stylesheet_count"],
        "console_lines": con["count"],
        "diff_summary": dom2["diff"]["summary"],
        "record_entries": final_rec["entries"],
        "reward": done.reward,
    }


def run_tool_roundtrip(client) -> dict:
    t = client.rpc(ToolCallRequest(tool_name="browser_navigate",
                                   arguments={"url": "https://shop.test/product/2"}),
                   timeout=30)
    assert isinstance(t, ToolCallResponse), type(t)
    payload = t.data.data
    assert payload["success"], payload.get("error")
    return {"navigated_to": payload["data"]["url"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8791")
    args = ap.parse_args()

    client = build_client(args.base_url)
    try:
        print("── external task bundle (client-side) ──────────")
        print("  tasks:", ", ".join(sorted(TASKS)))

        print("── tool surface ───────────────────────────────")
        tr = run_tool_roundtrip(client)
        print(f"  navigate -> {tr['navigated_to']}")

        print("── full episode over HTTP ─────────────────────")
        res = run_add_widget_to_cart(client)
        for k, v in res.items():
            print(f"  {k}: {v}")

        print("── capture surfaces over HTTP ─────────────────")
        cap = run_capture_surfaces(client)
        for k, v in cap.items():
            print(f"  {k}: {v}")

        print()
        ok = res["passed"] and tr["navigated_to"] and cap["reward"] == 1.0
        print("E2E RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        client.disconnect()


if __name__ == "__main__":
    sys.exit(main())