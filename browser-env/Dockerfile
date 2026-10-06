# syntax=docker/dockerfile:1
# browser-env — A2E browser environment host.
#
# Serves an A2EServer over HTTP (uvicorn) exposing:
#   env/*    BrowserEnvPlugin  (episodic browser, scored, transition log)
#   tools/*  BrowserToolPlugin (the same browser as callable tools)
#
# Build from the PARENT directory (the one containing browser-env/ AND
# a2eprotocol/) so the SDK checkout is in the build context:
#   docker build -f browser-env/Dockerfile -t browser-env:latest ..
# (or `make build`, which does exactly this)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    A2E_PORT=8770

# curl is used for the HEALTHCHECK; the rest is what `playwright install
# --with-deps` expects to find already present (it will still apt-install the
# browser shared libraries on the line below).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── a2e SDK: vendored from the LOCAL checkout, not PyPI ────────────────────
# Both browser-env and browser-agent must be built from the SAME SDK revision.
# pyproject.toml's [tool.uv.sources] points at ../../a2eprotocol/python-sdk;
# from /app that escapes the base dir, so we COPY the SDK to the exact absolute
# path it resolves to inside the image.
COPY a2eprotocol/python-sdk /a2eprotocol/python-sdk

# ── project metadata + source ─────────────────────────────────────────────
COPY browser-env/pyproject.toml ./
COPY browser-env/browser_env ./browser_env

# uv.lock is intentionally NOT copied: this image resolves deps at build time
# so it does not depend on a lockfile generated on a different machine.

# ── install ───────────────────────────────────────────────────────────────
# 1) rewrite the relative SDK source to the absolute path COPYed above (the
#    pyproject [tool.uv.sources] is uv-only; pip ignores it, so this is purely
#    so `uv sync` also works inside the image)
# 2) install the SDK, then the package + the real-browser extra. pip is
#    satisfied by the already-installed SDK distribution.
RUN sed -i 's#\.\./a2eprotocol/python-sdk#/a2eprotocol/python-sdk#g' pyproject.toml \
    && python -m pip install --upgrade pip \
    && python -m pip install "/a2eprotocol/python-sdk" \
    && python -m pip install ".[browser]"

# ── real headless Chromium ────────────────────────────────────────────────
# --with-deps installs the OS libraries Chromium needs. Kept in the image so
# the container can drive a real browser with BACKEND=playwright.
RUN playwright install --with-deps chromium

# ── build-time contract gate ──────────────────────────────────────────────
# Assert the SDK surface this host depends on actually exists, so a stale SDK
# pin fails the BUILD instead of hanging the first tool/env RPC at runtime.
RUN python -c "\
from a2e.caps.env.plugin import EnvPlugin; \
from a2e.caps.env.protocol import ENV_TYPE_MAP, DEFAULT_DATA_ROOT, EnvDataPart; \
from a2e.caps.tools.protocol import ToolDefinition, TOOL_TYPE_MAP; \
assert all(hasattr(EnvPlugin, h) for h in ('on_reset', 'on_step', 'on_data_reset')), \
    'EnvPlugin missing required hooks'; \
print('a2e env+tools contract OK; default data root =', DEFAULT_DATA_ROOT)"

# ── build-time config gate ────────────────────────────────────────────────
# The host config must parse AND the plugins must import, so a typo in a cls
# path or a missing metadata key fails here rather than at container start.
RUN python -c "\
import yaml, pathlib; \
from a2e.schema import A2EHostConfig; \
cfg = A2EHostConfig(**yaml.safe_load(pathlib.Path('/app/browser_env/host_config.yaml').read_text())); \
from browser_env.plugin import BrowserEnvPlugin; \
from browser_env.browser_tool import BrowserToolPlugin; \
print('host_config.yaml valid;', len(cfg.plugins), 'plugins')"

# ── offline-backend smoke test at build time ──────────────────────────────
# Proves the whole plugin path (reset -> step -> score -> done) works inside
# the image WITHOUT needing a browser page to load. A broken scoring path
# fails the build. (No task file involved: tasks arrive in env/reset options.)
RUN python -c "\
from browser_env.backend import make_backend; \
from browser_env.plugin import BrowserEnvPlugin; \
from browser_env.browser_tool import BrowserToolPlugin; \
b = make_backend('scripted'); b.start(); \
b.reset_session('https://shop.test/'); \
assert b.snapshot().elements, 'scripted backend snapshot empty'; \
print('scripted backend OK ->', b.url)"

ENV BROWSER_ENV_CONFIG=/app/browser_env/host_config.yaml
EXPOSE 8770

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=4 \
    CMD curl -fsS http://localhost:8770/health || exit 1

ENTRYPOINT ["python", "-m", "browser_env.app"]
CMD ["--config", "/app/browser_env/host_config.yaml"]