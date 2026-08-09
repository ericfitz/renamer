# Ollama test lifecycle: `pytest --ollama`

**Date:** 2026-08-08
**Status:** Approved, ready for planning

## Problem

`tests/test_llm_validation.py` holds the only four tests that exercise a live LLM
through LangChain. They skip unless an Ollama daemon is running:

```
SKIPPED [4] tests/test_llm_validation.py: Ollama is not running
```

Running them today is a manual three-step chore — start `ollama serve`, run
pytest, remember to stop the daemon. The "remember to stop it" step is the one
that gets skipped, leaving a daemon running indefinitely.

This matters more than a normal skip. These four tests are the only coverage
over the LangChain provider layer, which is exactly where dependency bumps land.
A green `59 passed, 4 skipped` can hide a broken LLM integration.

## Goal

One command runs everything and leaves the machine as it found it:

```
uv run pytest             # 59 passed, 4 skipped  (fast default, unchanged)
uv run pytest --ollama    # starts ollama, 63 passed, stops ollama
```

## Key constraint

The skip guard is a module-level `pytestmark` (`tests/test_llm_validation.py:52-55`),
so it is evaluated **at import time**. A session-scoped fixture would run too
late — the skip decision is already made by then.

`pytest_configure` runs before collection, and therefore before the module is
imported. Verified empirically, along with teardown reliability:

| Exit path | `pytest_unconfigure` runs? |
|---|---|
| Clean exit | yes |
| Test failure | yes |
| Collection error | yes |
| KeyboardInterrupt (Ctrl-C) | yes |

`SIGKILL` is the one path where teardown cannot run. Accepted: the orphaned
daemon is detected as already-running on the next run and left alone, so the
failure mode is a stray daemon, never a broken test run. A pidfile to close
this gap is deliberately out of scope.

## Architecture

Three units, each with one job.

### 1. `tests/ollama_support.py`

Plain Python; imports no pytest. Owns the model registry and daemon lifecycle,
so the safety-critical logic is testable without a pytest session.

```python
OLLAMA_ENDPOINT = "http://localhost:11434"

# config alias -> ollama model tag
OCR_MODEL      = ("llava-local",   "llava:13b")
ANALYSIS_MODEL = ("llama32-local", "llama3.2:3b")
REQUIRED_MODELS = [OCR_MODEL, ANALYSIS_MODEL]

def is_running() -> bool: ...
def missing_models() -> list[str]: ...
def start() -> subprocess.Popen: ...
def stop(proc: subprocess.Popen) -> None: ...
```

### 2. `tests/conftest.py`

Pytest wiring only, no business logic: `pytest_addoption` (registers
`--ollama`), `pytest_configure`, `pytest_unconfigure`, `pytest_report_header`.

### 3. `tests/test_llm_validation.py`

Minimal edits:
- Import the registry and build the `test_config_file` YAML from it.
- Delete the now-duplicate local `is_ollama_running()` / `OLLAMA_AVAILABLE`;
  use the shared helper.
- Add a comment naming the conftest hook, so the reason the import-time guard
  passes under `--ollama` is discoverable from the test side.

## Control flow

### Setup — `pytest_configure`, only when `--ollama` is passed

1. `shutil.which("ollama")` missing -> `UsageError` with an install hint.
2. Daemon already responding -> record `started_by_us = False`. **Do not stop it later.**
3. Otherwise spawn `ollama serve` detached; poll `/api/tags` until ready,
   60s timeout. On timeout: kill the child, then `UsageError`.
4. Any required model absent -> `UsageError` listing the exact `ollama pull`
   command per missing model. If we started the daemon in step 3, stop it
   before raising so we do not leak it.

### Teardown — `pytest_unconfigure`

Stop the daemon **only if `started_by_us`**: SIGTERM, wait with timeout,
SIGKILL fallback, confirm the port is closed. Otherwise no-op.

### Reporting — `pytest_report_header`

One line, either:
- `ollama: started for this session (will stop at exit)`
- `ollama: already running (left as-is)`

### Without the flag

conftest is inert. Today's skip behavior is completely unchanged.

### Interaction with marker selection

`--ollama` only makes the guard pass; it does not select tests. These tests keep
their `slow` marker, so `uv run pytest --ollama -m "not slow"` still deselects
them — and in that case the daemon is started for nothing. Acceptable: the
combination is contradictory on its face, and detecting it would mean
second-guessing an explicit flag.

## Decisions and rationale

**Fail fast on missing models; never auto-download.** A bare `--ollama` must
never kick off an ~8GB `llava:13b` pull. The error names the exact command; the
cost is one-time per machine. A separate `--ollama-pull` flag was considered and
rejected as YAGNI — a second flag and its tests to save typing one command once.

**`UsageError`, not skip.** An explicit `--ollama` that cannot be honored should
fail loudly. Silently skipping would reproduce the exact problem this solves.

**Never stop a daemon we did not start.** Killing a long-running daemon the
developer started themselves is a hostile surprise.

**Pin `llama3.2:3b`, not `llama3.2:latest`.** These tests make subjective
quality assertions judged by a second LLM, so a model changing underneath them
flips results with no code change. `llama3.2:3b` and `llama3.2:latest` currently
resolve to the identical digest (`sha256:34bb5ab…`), already present on disk, so
this costs no download and changes no behavior today — it only stops `latest`
from moving later. `llava:13b` is already a concrete size tag.

Digest verification (recording expected digests, warning on mismatch) was
considered and rejected: concrete size tags are stable enough.

**Single source for the model list.** The tags are currently hardcoded inside
the `test_config_file` fixture. A second copy in the preflight would be free to
drift — the same duplicate-declaration bug class fixed in `0ebc057`. The
registry in `ollama_support.py` is the one definition; the YAML fixture is built
from it.

## Testing

New `tests/test_ollama_support.py`, all mocked — no unit test spawns a real
daemon:

- `missing_models()` correctly diffs the registry against the API tag list
  (none missing / one missing / all missing).
- Teardown does **not** stop a daemon it did not start. *(safety-critical)*
- Teardown **does** stop one it did start.
- Preflight raises `UsageError` naming the right `ollama pull` command.

Manual verification:
- `uv run pytest` -> 59 passed, 4 skipped; no daemon started.
- `uv run pytest --ollama` -> 63 passed; daemon stopped afterward, port closed.
- `uv run pytest --ollama` with the daemon already up -> 63 passed; daemon
  still running afterward.

## Out of scope

- CI wiring (the repo has no CI).
- Auto-pulling models.
- Pidfile recovery for `SIGKILL`.
- The `~/Documents` corpus dependency in the `documents_dir` fixture.
- Extending live coverage to hosted providers (anthropic / openai / google /
  xai), which needs API keys. Tracked separately as a known coverage gap.
