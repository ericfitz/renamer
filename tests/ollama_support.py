#!/usr/bin/env python3
"""Ollama model registry and daemon lifecycle for the test suite.

Imports no pytest: the lifecycle decisions live here so they can be tested
directly, and `tests/conftest.py` is left as thin pytest wiring.

The one rule worth stating twice: we stop the daemon only if we started it.
A developer who already had `ollama serve` running keeps it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Callable, Iterable, Sequence

OLLAMA_ENDPOINT = "http://localhost:11434"

# Single source of truth for the models these tests need.
#
# (alias used in the generated test config, ollama model tag)
#
# Tags are pinned to concrete sizes rather than ":latest". The LLM validation
# tests assert on subjective quality ("is this name sensible?"), so a model
# changing underneath them looks like a code regression when it is not.
OCR_MODEL = ("llava-local", "llava:13b")
ANALYSIS_MODEL = ("llama32-local", "llama3.2:3b")
REQUIRED_MODELS = [OCR_MODEL, ANALYSIS_MODEL]

STARTUP_TIMEOUT_SECONDS = 60.0
POLL_INTERVAL_SECONDS = 0.5
SHUTDOWN_TIMEOUT_SECONDS = 10.0


class OllamaStartupError(RuntimeError):
    """Raised when a daemon we spawned never became ready."""


def missing_models(
    installed: Iterable[str],
    required: Sequence[tuple[str, str]] = REQUIRED_MODELS,
) -> list[str]:
    """Return the required model tags absent from `installed`, in declared order."""
    have = set(installed)
    return [tag for _alias, tag in required if tag not in have]


def pull_commands(tags: Iterable[str]) -> list[str]:
    """Return the exact `ollama pull` command for each missing tag."""
    return [f"ollama pull {tag}" for tag in tags]


def is_installed() -> bool:
    """True if the ollama CLI is on PATH."""
    return shutil.which("ollama") is not None


def is_running(endpoint: str = OLLAMA_ENDPOINT, timeout: float = 2.0) -> bool:
    """True if a daemon answers on `endpoint`."""
    try:
        with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def installed_models(endpoint: str = OLLAMA_ENDPOINT, timeout: float = 10.0) -> set[str]:
    """Return the model tags the running daemon reports."""
    with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=timeout) as response:
        payload = json.load(response)
    return {model["name"] for model in payload.get("models", [])}


def spawn_daemon() -> subprocess.Popen:
    """Start `ollama serve` detached, so a Ctrl-C aimed at pytest does not race us."""
    return subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def terminate_daemon(proc: subprocess.Popen) -> None:
    """Stop a daemon we spawned: SIGTERM, then SIGKILL if it will not go."""
    proc.terminate()
    try:
        proc.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)


def wait_until_ready(
    probe: Callable[[], bool] = is_running,
    timeout: float = STARTUP_TIMEOUT_SECONDS,
    interval: float = POLL_INTERVAL_SECONDS,
) -> bool:
    """Poll `probe` until it is true or `timeout` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe():
            return True
        time.sleep(interval)
    return probe()


class OllamaSession:
    """Owns the 'did we start it?' decision, and therefore the right to stop it."""

    def __init__(
        self,
        *,
        probe: Callable[[], bool] = is_running,
        spawn: Callable[[], subprocess.Popen] = spawn_daemon,
        terminate: Callable[[subprocess.Popen], None] = terminate_daemon,
        wait_ready: Callable[[], bool] = wait_until_ready,
    ):
        self._probe = probe
        self._spawn = spawn
        self._terminate = terminate
        self._wait_ready = wait_ready
        self._proc: subprocess.Popen | None = None

    @property
    def started_by_us(self) -> bool:
        return self._proc is not None

    def ensure_running(self) -> None:
        """Start a daemon if none is answering. No-op when one already is."""
        if self._probe():
            return
        proc = self._spawn()
        if not self._wait_ready():
            # Never leave behind a daemon that failed to come up.
            self._terminate(proc)
            raise OllamaStartupError(
                f"ollama serve did not become ready within {STARTUP_TIMEOUT_SECONDS:.0f}s"
            )
        self._proc = proc

    def shutdown(self) -> None:
        """Stop the daemon only if this session started it."""
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        self._terminate(proc)
