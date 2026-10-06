# browser-env

A2E environment host for browser / computer-use agents. Serves an A2EServer over
HTTP and exposes a real (headless Chromium) or deterministic (scripted) browser
under two capabilities:

| capability | namespace | what it is for |
|---|---|---|
| `env`   | `env/*`   | episodic RL surface: `reset(task)` → `step(action)` → `(obs, reward, done)`, scored, replayable, logged as transitions |
| `tools` | `tool/*`  | inference surface: `browser_navigate`, `browser_click`, … callable as ordinary tools |

Only **SDK-defined** message types are used (`a2eprotocol/python-sdk`): this
host defines no custom `.../req`/`.../resp` pairs and no `A2EMessage`
subclasses. Score rides the terminal observation + `EnvStatePush` reward, and
the `browser_score` tool (`tool/call`).

Both delegate to **one backend and one episode record**, so the two surfaces can
never disagree about where the browser is. A tool call goes through the same
`step()` as an env step: it consumes an episode step, applies the same reward,
and lands in the same experience log.

The agent side lives in the sibling repo `browser-agent/`. The two share **only
the a2e protocol** — this repo is never imported by the agent.

## Install / run

```bash
make venv          # uv venv + local a2e SDK checkout + this package
make test          # offline suite (scripted backend), no browser needed

python scripts/make_local_config.py            # writes host_config.local.yaml
python -m browser_env.app --config browser_env/host_config.local.yaml

python scripts/e2e_http.py --base-url http://localhost:8791   # full wire check
```

Container:

```bash
make build         # context = repo root (so a2eprotocol/ is in reach)
make run
```

## Action space

Every browser action is one `env/step`. Anything outside this list is rejected
with `unknown action` rather than silently ignored.

| action | payload | notes |
|---|---|---|
| `navigate` | `url` | |
| `click` | `selector` \| `index` | `index` is the stable `data-a2e-eid` from the snapshot |
| `fill` | `selector`/`index`, `text` | |
| `press` | `key`, `selector`/`index` | |
| `scroll` | `direction` (`down\|up\|top\|bottom`), `amount` | |
| `back` / `forward` / `reload` | — | |
| `extract` | `selector`/`index` | visible text |
| `screenshot` | `full_page` | base64 PNG in `metadata.render` |
| **`dom`** | `mode` (`capture\|diff`), `max_bytes` | style-flattened DOM, below |
| **`console`** | `max_entries`, `clear` | page console output + uncaught JS errors |
| **`record`** | `mode` (`start\|stop\|status`), `include_screenshots` | JSONL trajectory artifact |
| `finish` | `answer` | terminal: scores the episode |

Actions that fail (bad selector, unknown mode, no baseline) do **not** kill the
episode — the error comes back in `state.last_error` with `metadata.action_ok
= False` so the agent can correct course. Stepping an already-finished episode
is an SDK-level error by design.

## Capture surfaces

These four exist because an agent driving a page it cannot see needs to know
*what happened*, not just what it did.

### `dom` — style-flattened DOM, diffable

Not a raw `outerHTML` dump. The capture carries everything a diff needs:

- **markup with COMPUTED styles written inline** (flattened onto a *clone* —
  the live DOM is never mutated), so two captures differ whenever any resolved
  property changed, even if only an external stylesheet moved;
- **every stylesheet** (`<style>` rules *and* link/external cssText, with a
  `readable: false` marker for cross-origin rules rather than a silent empty);
- the **interactive-element index** (`data-a2e-eid`), so element indexes survive
  into the captured markup;
- a **`checksum`** (`sha256:` over url + title + markup + all CSS) so equality
  is a string compare.

```python
# capture
step(episode, {"action_type": "dom", "payload": {}})   # -> metadata.dom

# after an action: did the page change? (no new wire type needed)
step(episode, {"action_type": "dom", "payload": {"mode": "diff"}})
# -> metadata.dom.diff = {changed, checksum_equal, summary, added_elements,
#                         removed_elements, stylesheet_changed,
#                         interactive_changed, unified}
```

The diff helper lives in `browser_env/domdiff.py`
(`checksum` / `diff_captures` / `has_changed`).

### `console` — what the page said

`console.*` messages **and** uncaught page errors (`pageerror`), episode-scoped
(`reset` clears it). This is usually where "my click did nothing" is answered.

Two ways to read it:
- explicit `console` action → full buffer in `metadata.console.entries`
- **automatic delta**: any step that produced new lines gets them in
  `metadata.console.new` (omitted entirely when empty, so a quiet step costs
  nothing). The JSONL recording stores the same delta per step.

`clear: true` empties the buffer and resets the delta cursor.

### `screenshot` — pixels

Base64 PNG in `metadata.render` (`{encoding, mime, bytes, data, omitted}`).
Omitted (with `omitted: true`) when over `SCREENSHOT_MAX_BYTES` rather than
pushing an unbounded blob onto the wire.

### `record` — the trajectory artifact

JSONL at `<ROOT>/recordings/<stamp>-<episode>.jsonl`, one row per step:
`{step, action, args, url, title, ok, error, reward, done, ts, console?}`.

- `mode: start` (optional `include_screenshots` to embed a PNG frame per step)
- `mode: status` / `mode: stop`
- **finalized automatically** when the episode ends — the terminal observation
  reports the artifact in `metadata.record`, so the agent does not have to
  remember to stop it.

Separate from the SQLite `ExperienceStore` on purpose: that one is the
`(s, a, r, s', done)` log for training, this is the human/debug replay. Different
consumers, different formats.

## Tasks live OUTSIDE the environment

The host keeps **no task catalogue**. The caller supplies the full task spec
in `env/reset` options (`options.task` = `{name, start_url, goal, max_steps?,
success?, ...}`); a bare task NAME is rejected with an explicit error because
there is nothing to resolve it against. A typo'd criterion key fails at reset
rather than being silently unchecked.

The reference dataset sits OUTSIDE both packages, as a sibling directory:

```
browser-env/
├── browser_env/     the host package
├── browser_tasks/   the task DATASET (tasks.json) — loaded by callers only
```

Callers (`scripts/e2e_http.py`, `tests/`, and the agent's local copy) read the
dataset and pass the whole spec; the plugin never opens a task file.

## Scoring

Scoring is `score_episode(task, record)` → `score ∈ [0,1]` = fraction of
declared criteria met, plus per-criterion verdicts. It runs on the **host's**
own episode record:

- terminal `finish` puts `score` / `passed` / `criteria` in the observation
  `metadata`, and the terminal reward is pushed as an `EnvStatePush`;
- the `browser_score` tool re-derives it on demand via SDK `tool/call`
  (optional caller-supplied `record` hints are filled in ONLY where the host
  record has no value — the host's view overwrites them, so a caller cannot
  self-report success).

Reward: step shaping (`PROGRESS_REWARD` / `STEP_PENALTY`, both 0 by default) and
the terminal reward = the episode score. The terminal reward is pushed as an
`EnvStatePush`.

## Configuration

`browser_env/host_config.yaml` is a pure `A2EHostConfig` — no agent-side keys
(they would be a startup `ValidationError`). It uses absolute `/app/...` paths
for the container; generate a host-side copy with `scripts/make_local_config.py`.

Required plugin metadata (the plugin fails at startup, not at first use):
`BACKEND`, `ROOT`. There is deliberately no task-file key.

| key | default | meaning |
|---|---|---|
| `BACKEND` | — (required) | `playwright` \| `scripted` |
| `ROOT` | — (required) | experience store + recordings |
| `DEFAULT_MAX_STEPS` | 15 | ad-hoc episode budget |
| `MAX_TEXT_CHARS` / `MAX_ELEMENTS` | 4000 / 120 | snapshot size |
| `DOM_MAX_BYTES` | 500000 | DOM capture cap |
| `CONSOLE_MAX_ENTRIES` | 40 | per-step console delta cap |
| `SCREENSHOT_MAX_BYTES` | 1500000 | screenshot cap |
| `STEP_PENALTY` / `PROGRESS_REWARD` | 0 / 0 | step shaping |

## Backends

- **`playwright`** — real headless Chromium (`make build` installs it). Lazy
  import: the package works without it installed.
- **`scripted`** — an in-memory model of a small shop (`https://shop.test/`),
  dependency- and network-free. It implements the *full* contract including
  stylesheets, computed styles, console output and DOM, so tests written against
  it also run against Chromium.

```bash
python -m pytest -q        # 47 tests, offline
```

## Layout

The task dataset lives beside the package, never inside it:
`browser_tasks/tasks.json` (loaded by callers — the host never opens it).

```
browser_env/
├── actions.py       action vocabulary, payload normalization, target resolution
├── plugin.py        BrowserEnvPlugin  (env/* capability)
├── browser_tool.py  BrowserToolPlugin (tool/* capability, same browser)
├── backend/
│   ├── base.py                BrowserBackend contract + snapshot types
│   ├── playwright_backend.py  real Chromium
│   └── scripted_backend.py    offline deterministic model
├── domdiff.py       checksum / diff_captures / has_changed
├── tasking.py       BrowserTask spec validation + deterministic scoring
├── state.py         BrowserEpisode, ExperienceStore (SQLite), EpisodeRecorder (JSONL)
├── host_config.yaml container config (pure A2EHostConfig)
└── app.py           `python -m browser_env.app --config <yaml>`
```

(no `protocol.py`: message types come from the SDK only)