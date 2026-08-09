#!/usr/bin/env python3
"""Unit tests for the Ollama test-lifecycle helpers.

No test here starts a real daemon or touches the network; the collaborators
are injected so the decisions can be exercised directly.
"""

import pytest

from tests.ollama_support import (
    REQUIRED_MODELS,
    OllamaSession,
    OllamaStartupError,
    missing_models,
    pull_commands,
)


class FakeProc:
    """Stand-in for the daemon subprocess."""

    def __init__(self, name="ollama"):
        self.name = name


class TestMissingModels:
    def test_returns_empty_when_every_required_model_is_installed(self):
        installed = {tag for _alias, tag in REQUIRED_MODELS}
        assert missing_models(installed) == []

    def test_reports_the_model_that_is_absent(self):
        required = [("ocr", "llava:13b"), ("analysis", "llama3.2:3b")]
        assert missing_models({"llama3.2:3b"}, required=required) == ["llava:13b"]

    def test_reports_every_model_when_none_are_installed(self):
        required = [("ocr", "llava:13b"), ("analysis", "llama3.2:3b")]
        assert missing_models(set(), required=required) == ["llava:13b", "llama3.2:3b"]

    def test_ignores_unrelated_installed_models(self):
        required = [("analysis", "llama3.2:3b")]
        assert missing_models({"qwen3-coder:latest", "gemma3:latest"}, required=required) == [
            "llama3.2:3b"
        ]


class TestPullCommands:
    def test_names_the_exact_command_for_each_missing_model(self):
        assert pull_commands(["llava:13b", "llama3.2:3b"]) == [
            "ollama pull llava:13b",
            "ollama pull llama3.2:3b",
        ]


class TestSessionLeavesForeignDaemonAlone:
    """The safety-critical behavior: never stop a daemon we did not start."""

    def test_does_not_spawn_when_a_daemon_is_already_running(self):
        spawned = []
        session = OllamaSession(
            probe=lambda: True,
            spawn=lambda: spawned.append("spawn") or FakeProc(),
            terminate=lambda proc: None,
            wait_ready=lambda: True,
        )

        session.ensure_running()

        assert spawned == []
        assert session.started_by_us is False

    def test_shutdown_does_not_terminate_a_daemon_it_did_not_start(self):
        terminated = []
        session = OllamaSession(
            probe=lambda: True,
            spawn=lambda: FakeProc(),
            terminate=terminated.append,
            wait_ready=lambda: True,
        )
        session.ensure_running()

        session.shutdown()

        assert terminated == []

    def test_shutdown_is_a_noop_when_ensure_running_was_never_called(self):
        terminated = []
        session = OllamaSession(
            probe=lambda: False,
            spawn=lambda: FakeProc(),
            terminate=terminated.append,
            wait_ready=lambda: True,
        )

        session.shutdown()

        assert terminated == []


class TestSessionStopsWhatItStarted:
    def test_spawns_when_no_daemon_is_running(self):
        proc = FakeProc()
        session = OllamaSession(
            probe=lambda: False,
            spawn=lambda: proc,
            terminate=lambda p: None,
            wait_ready=lambda: True,
        )

        session.ensure_running()

        assert session.started_by_us is True

    def test_shutdown_terminates_the_daemon_it_started(self):
        proc = FakeProc()
        terminated = []
        session = OllamaSession(
            probe=lambda: False,
            spawn=lambda: proc,
            terminate=terminated.append,
            wait_ready=lambda: True,
        )
        session.ensure_running()

        session.shutdown()

        assert terminated == [proc]

    def test_shutdown_twice_terminates_only_once(self):
        terminated = []
        session = OllamaSession(
            probe=lambda: False,
            spawn=lambda: FakeProc(),
            terminate=terminated.append,
            wait_ready=lambda: True,
        )
        session.ensure_running()

        session.shutdown()
        session.shutdown()

        assert len(terminated) == 1


class TestStartupFailure:
    def test_terminates_the_spawned_process_when_it_never_becomes_ready(self):
        proc = FakeProc()
        terminated = []
        session = OllamaSession(
            probe=lambda: False,
            spawn=lambda: proc,
            terminate=terminated.append,
            wait_ready=lambda: False,
        )

        with pytest.raises(OllamaStartupError):
            session.ensure_running()

        assert terminated == [proc], "a daemon that never came up must not be leaked"

    def test_does_not_claim_ownership_after_a_failed_start(self):
        session = OllamaSession(
            probe=lambda: False,
            spawn=lambda: FakeProc(),
            terminate=lambda p: None,
            wait_ready=lambda: False,
        )

        with pytest.raises(OllamaStartupError):
            session.ensure_running()

        assert session.started_by_us is False
