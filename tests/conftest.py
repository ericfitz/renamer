#!/usr/bin/env python3
"""Pytest wiring for the optional Ollama test lifecycle.

    uv run pytest             # LLM validation tests skip (fast default)
    uv run pytest --ollama    # start ollama, run everything, stop ollama

All decisions live in `tests/ollama_support.py`; this file only connects them
to pytest hooks.

Why `pytest_configure` and not a fixture: the guard in test_llm_validation.py
is a module-level `pytestmark`, so it is evaluated when that module is
imported. `pytest_configure` runs before collection, and therefore before the
import, which is the only place early enough to make the guard see a live
daemon. A session-scoped fixture would run after the skip decision was made.
"""

from __future__ import annotations

import pytest

from tests import ollama_support
from tests.ollama_support import OllamaSession, OllamaStartupError

_SESSION_ATTR = "_ollama_session"


def pytest_addoption(parser):
    parser.addoption(
        "--ollama",
        action="store_true",
        default=False,
        help="Start Ollama for the LLM validation tests, then stop it again afterwards. "
        "A daemon that was already running is left alone.",
    )


def pytest_configure(config):
    if not config.getoption("--ollama"):
        return

    if not ollama_support.is_installed():
        raise pytest.UsageError(
            "--ollama was given but the 'ollama' command is not on PATH.\n"
            "  Install it from https://ollama.com/download"
        )

    session = OllamaSession()
    try:
        session.ensure_running()
    except OllamaStartupError as exc:
        raise pytest.UsageError(f"--ollama: {exc}") from exc

    try:
        missing = ollama_support.missing_models(ollama_support.installed_models())
    except OSError as exc:
        session.shutdown()
        raise pytest.UsageError(f"--ollama: could not list installed models: {exc}") from exc

    if missing:
        # Report before tearing down, and never start a multi-GB download
        # the caller did not ask for.
        session.shutdown()
        commands = "\n".join(f"  {cmd}" for cmd in ollama_support.pull_commands(missing))
        raise pytest.UsageError(
            "--ollama: required model(s) not available:\n"
            f"{commands}\n"
            "Pull them and re-run."
        )

    setattr(config, _SESSION_ATTR, session)


def pytest_report_header(config):
    session = getattr(config, _SESSION_ATTR, None)
    if session is None:
        return None
    if session.started_by_us:
        return "ollama: started for this session (will stop at exit)"
    return "ollama: already running (left as-is)"


def pytest_unconfigure(config):
    session = getattr(config, _SESSION_ATTR, None)
    if session is not None:
        session.shutdown()
