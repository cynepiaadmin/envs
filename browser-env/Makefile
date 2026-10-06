# ─────────────────────────────────────────────────────────────────────────
# browser-env — build / run the A2E browser environment host.
#
# The build context is the PARENT directory (the one holding browser-env/ and
# a2eprotocol/), so the a2e SDK checkout is reachable inside the image. Both
# browser-env and browser-agent MUST share one context, because both pin the
# SDK from the same local checkout — building from browser-env/ alone would put
# the SDK out of reach and silently fall back to whatever was on PyPI.
# ─────────────────────────────────────────────────────────────────────────

SHELL       := /bin/bash
APP_NAME    := browser-env
IMAGE_NAME  := browser-env
VERSION_TAG := latest
REGISTRY    ?=
PORT        ?= 8770

# Anchor every path to the directory holding THIS Makefile, so `make build`
# works whether it is run from browser-env/ or from the parent via
# `make -f browser-env/Makefile`. `-f` is resolved against the invocation
# cwd, NOT the build context — a relative DOCKERFILE breaks one of the two
# ways of invoking it.
MAKEFILE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO_ROOT    := $(abspath $(MAKEFILE_DIR)/..)
DOCKERFILE   := $(MAKEFILE_DIR)/Dockerfile

.PHONY: all build run stop clean rebuild config test test-local venv help

all: build

build: ## Build the image (context = repo root)
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME):$(VERSION_TAG) $(REPO_ROOT)

tag: ## Tag for a registry push (make tag REGISTRY=registry.example.com)
ifdef REGISTRY
	docker tag $(IMAGE_NAME):$(VERSION_TAG) $(REGISTRY)/$(IMAGE_NAME):$(VERSION_TAG)
else
	@echo "Usage: make tag REGISTRY=registry.example.com"
endif

push: tag ## Push the tagged image
ifdef REGISTRY
	docker push $(REGISTRY)/$(IMAGE_NAME):$(VERSION_TAG)
else
	@echo "Usage: make push REGISTRY=registry.example.com"
endif

run: ## Run the env host container on $(PORT)
	docker run --rm -p $(PORT):8770 --name $(APP_NAME) $(IMAGE_NAME):$(VERSION_TAG)

stop: ## Stop a running container
	-docker stop $(APP_NAME) 2>/dev/null; docker rm $(APP_NAME) 2>/dev/null; true

clean: ## Remove the image
	-docker rmi $(IMAGE_NAME):$(VERSION_TAG) 2>/dev/null; true

rebuild: clean build ## Clean then build

config: ## Generate a host-side host_config.local.yaml (absolute paths, scripted backend by default)
	cd $(MAKEFILE_DIR) && python3 scripts/make_local_config.py

venv: ## Create a local venv with the SDK + this package installed (dev)
	cd $(MAKEFILE_DIR) && uv venv .venv --python 3.11
	cd $(MAKEFILE_DIR) && uv pip install --python .venv/bin/python -e $(REPO_ROOT)/a2eprotocol/python-sdk
	cd $(MAKEFILE_DIR) && uv pip install --python .venv/bin/python -e ".[dev]"

test: ## Run the (offline, scripted-backend) test suite
	cd $(MAKEFILE_DIR) && pytest -q

test-local: ## Same, via the local venv
	cd $(MAKEFILE_DIR) && .venv/bin/pytest -q

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'