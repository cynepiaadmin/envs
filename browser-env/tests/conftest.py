"""In-process A2E host fixture: a real client↔server RPC pair, no network.

The agent repo has a subprocess-based host fixture. Here we use the SDK's
DirectTransport, which is a genuine protocol path (encode → queue → decode →
dispatch) minus the socket. That is fast, hermetic, and still exercises every
message the wire would carry.

Why this matters: the #1 failure mode in this family is a client/server
TYPE_MAP mismatch, which does NOT raise — the RPC just hangs until timeout.
Driving a real Transport pair is what catches it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from a2e.core.client.client import A2EClient
from a2e.core.server.executor import A2EServerRuntimeExecutor
from a2e.core.transports.direct import DirectTransport
from a2e.schema import A2EHostConfig

from a2e.caps.env.protocol import ENV_TYPE_MAP
from a2e.caps.tools.protocol import TOOL_TYPE_MAP

from browser_env.browser_tool import BrowserToolPlugin
from browser_env.plugin import BrowserEnvPlugin
from browser_env.browser_tool import BrowserToolPlugin
# Capabilities the test agent advertises. The executor only accepts a plugin
# whose config['type'] appears here.
AGENT_CAPS = ["env", "tools"]


def host_config_dict(root: Path, backend: str = "scripted",
                     default_url: str = "") -> dict:
    """Build a validated-shape A2EHostConfig dict for tests."""
    return {
        "host_id": "browser-env-test",
        "transport": {
            "type": "http",
            "config": {"base_url": "http://localhost:8770",
                       "send_path": "/send", "stream_path": "/stream"},
        },
        "server": {"host": "127.0.0.1", "port": 0},
        "auth_token": None,
        "audit": {"enabled": False, "path": None,
                  "session_id_source": "host_id"},
        "snapshot_store": {"type": "file", "config": {"root": str(root)}},
        "plugins": [
            {
                "name": "browser_env",
                "type": "env",
                "cls": "browser_env.plugin.BrowserEnvPlugin",
                "metadata": {
                    "enabled": True, "priority": 0, "exclusive": False,
                    "BACKEND": backend,
                    "ROOT": str(root),
                    "DEFAULT_MAX_STEPS": 8,
                    "MAX_TEXT_CHARS": 2000,
                    "MAX_ELEMENTS": 40,
                },
            },
            {
                "name": "browser_tools",
                "type": "tools",
                "cls": "browser_env.browser_tool.BrowserToolPlugin",
                "metadata": {
                    "enabled": True, "priority": 5, "exclusive": False,
                    "ENV_PLUGIN": "browser_env",
                    "DEFAULT_URL": default_url,
                    "MAX_TEXT_CHARS": 2000,
                },
            },
        ],
    }


class Host:
    """A started host + a connected client."""

    def __init__(self, executor, client, config):
        self.executor = executor
        self.client = client
        self.config = config

    @property
    def env_plugin(self) -> BrowserEnvPlugin:
        return self.executor.get_plugin("browser_env")

    @property
    def tool_plugin(self) -> BrowserToolPlugin:
        return self.executor.get_plugin("browser_tools")

    def close(self) -> None:
        try:
            self.client.disconnect()
        except Exception:
            pass
        try:
            self.executor.stop()
        except Exception:
            pass


@pytest.fixture
def make_host(tmp_path):
    """Factory fixture: `make_host(backend="scripted", default_url=...)`."""
    started: list[Host] = []

    def _make(backend: str = "scripted", default_url: str = "",
              agent_caps: list[str] | None = None) -> Host:
        logger = logging.getLogger(f"a2e-test-{len(started)}")
        logger.setLevel(logging.CRITICAL)
        root = tmp_path / f"root-{len(started)}"
        root.mkdir(parents=True, exist_ok=True)

        cfg = A2EHostConfig(**host_config_dict(root, backend, default_url))

        t_server = DirectTransport(logger=logger)
        t_client = DirectTransport(logger=logger)
        t_server.connect(t_client)

        executor = A2EServerRuntimeExecutor(cfg, t_server, logger)
        executor.start()

        client = A2EClient(
            transport=t_client,
            logger=logger,
            agent_id="browser-agent-test",
            agent_caps=agent_caps or AGENT_CAPS,
        )
        # Register EVERY namespace the client consumes on the CLIENT side too.
        # A2EClient only seeds A2E_BASE_TYPE_MAP: without these updates the
        # response decodes to a bare A2EMessage (req_id is dropped), never
        # matches the pending RPC, and the call HANGS until timeout with no
        # error anywhere. This is the SDK's #1 silent-hang cause.
        client.update_msg_types(ENV_TYPE_MAP)
        client.update_msg_types(TOOL_TYPE_MAP)
        client.connect()

        host = Host(executor, client, cfg)
        started.append(host)
        return host

    yield _make

    for h in started:
        h.close()


@pytest.fixture
def host(make_host) -> Host:
    return make_host()


__all__ = ["Host", "make_host", "host", "host_config_dict", "AGENT_CAPS"]