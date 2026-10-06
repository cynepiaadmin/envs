"""Standalone launcher for the browser environment A2E host.

Mirrors xa-agent-env's app.py: build a validated A2EHostConfig from a YAML file
and serve it over HTTP with uvicorn. This is the container entrypoint.

    python -m browser_env.app --config /app/browser_env/host_config.yaml

The config path is an explicit input — there is no silent fallback to a
different file, because the host config determines which browser backend runs.
"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from a2e import A2EServer
from a2e.schema import A2EHostConfig


def build_app(config_path: str = None, *, overrides: dict = None):
    """Return ``(asgi_app, validated_config)``."""
    if not config_path:
        raise ValueError("build_app requires config_path")
    import yaml
    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        _deep_merge(raw, overrides)
    config = A2EHostConfig(**raw)
    server = A2EServer(config=config, logger=logging.getLogger("browser_env"))
    return server.start(), config


def _deep_merge(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="A2E browser environment host")
    ap.add_argument("--config", required=True,
                    help="path to browser_env/host_config.yaml")
    args = ap.parse_args()

    overrides = {}
    if os.getenv("A2E_PORT"):
        overrides["server"] = {"port": int(os.environ["A2E_PORT"])}
    app, config = build_app(args.config, overrides=overrides)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()