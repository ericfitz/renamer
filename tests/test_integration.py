#!/usr/bin/env python3
"""
Integration tests for the renamer application.

Tests the application end-to-end using --dry-run mode to avoid LLM calls.
Uses real files from $HOME/Documents/ for realistic testing.

Run with: uv run pytest tests/test_integration.py -v
"""

import subprocess
from pathlib import Path

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def documents_dir():
    """Return the user's Documents directory."""
    docs = Path.home() / "Documents"
    if not docs.exists():
        pytest.skip("~/Documents directory does not exist")
    return docs


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_config_file(project_root):
    """Create a temporary config file for testing."""
    config_content = """
models:
  - name: llava-local
    provider: ollama
    endpoint: http://localhost:11434
    model: llava:13b

  - name: llama32-local
    provider: ollama
    endpoint: http://localhost:11434
    model: llama3.2:latest

configurations:
  - name: local-only
    description: All local models via Ollama
    ocr_model: llava-local
    analysis_model: llama32-local
    organizer_model: llama32-local

default_configuration: local-only
"""
    config_path = project_root / "test_config.yaml"
    config_path.write_text(config_content)
    yield config_path
    # Cleanup
    if config_path.exists():
        config_path.unlink()


def cleanup_output_files():
    """Remove any output files created during tests."""
    desktop = Path.home() / "Desktop"
    for pattern in ["file_organization_plan_*.md", "apply_changes_*.py", "revert_changes_*.py"]:
        for f in desktop.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass


# =============================================================================
# CLI Help Tests
# =============================================================================


class TestCLIHelp:
    """Test CLI help and basic invocation."""

    def test_help_flag(self, project_root):
        """Test that --help works."""
        result = subprocess.run(
            ["uv", "run", "renamer.py", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode == 0
        assert "file renamer and organizer" in result.stdout.lower()
        # CLI now has subcommands
        assert "run" in result.stdout
        assert "init-config" in result.stdout

    def test_run_command_help(self, project_root):
        """Test that run --help shows all expected options."""
        result = subprocess.run(
            ["uv", "run", "renamer.py", "run", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode == 0
        assert "--dry-run" in result.stdout
        assert "--include" in result.stdout
        assert "--exclude" in result.stdout
        assert "--recurse" in result.stdout
        assert "--max-files" in result.stdout
        assert "--config" in result.stdout
        assert "--profile" in result.stdout
        assert "--resume" in result.stdout
        assert "--auto-continue" in result.stdout

    def test_init_config_help(self, project_root):
        """Test that init-config --help works."""
        result = subprocess.run(
            ["uv", "run", "renamer.py", "init-config", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode == 0
        assert "configuration file" in result.stdout.lower()


# =============================================================================
# Dry Run Tests with ~/Documents
# =============================================================================


class TestDryRunWithDocuments:
    """Test dry-run mode with real files from ~/Documents."""

    def test_discovers_files_in_documents(self, documents_dir, project_root, test_config_file):
        """Test that dry-run discovers files in ~/Documents."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "10",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        assert result.returncode == 0
        assert "Pass 1" in result.stdout
        assert "Discovered" in result.stdout

        cleanup_output_files()

    def test_dry_run_completes_all_passes(self, documents_dir, project_root, test_config_file):
        """Test that dry-run completes all three passes."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "5",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        assert result.returncode == 0
        assert "Pass 1" in result.stdout
        assert "Pass 2" in result.stdout
        assert "Pass 3" in result.stdout
        assert "Processing Complete" in result.stdout

        cleanup_output_files()

    def test_dry_run_excludes_hidden_files(self, documents_dir, project_root, test_config_file):
        """Test that hidden files are excluded by default."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "20",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        assert result.returncode == 0
        # Hidden files like .DS_Store should not appear
        assert ".DS_Store" not in result.stdout
        assert ".hidden" not in result.stdout

        cleanup_output_files()

    def test_dry_run_with_include_pattern_pdf(self, documents_dir, project_root, test_config_file):
        """Test include pattern filtering for PDFs."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
                "--no-recurse",
                "--include",
                r".*\.pdf$",
                "--max-files",
                "10",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        assert result.returncode == 0
        # If any files are found, they should only be PDFs
        if "Discovered 0 files" not in result.stdout:
            # Check that non-PDFs don't appear in the file list
            lines = result.stdout.split("\n")
            for line in lines:
                if "│" in line and "." in line:  # Table row with file
                    # Should not contain non-PDF extensions
                    assert ".txt" not in line.lower() or ".pdf" in line.lower()

        cleanup_output_files()

    def test_dry_run_with_max_files_limit(self, documents_dir, project_root, test_config_file):
        """Test max files limit is respected."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "3",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        assert result.returncode == 0
        # Should discover at most 3 files
        assert (
            "Discovered 3 files" in result.stdout
            or "Discovered 2 files" in result.stdout
            or "Discovered 1 files" in result.stdout
            or "Discovered 0 files" in result.stdout
            or "No files found" in result.stdout
        )

        cleanup_output_files()

    def test_dry_run_creates_output_files(self, documents_dir, project_root, test_config_file):
        """Test that dry-run creates output files on Desktop."""
        # Clean up any existing files first
        cleanup_output_files()

        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "3",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        assert result.returncode == 0

        # Check that output files are mentioned
        assert "file_organization_plan" in result.stdout
        assert "apply_changes" in result.stdout
        assert "revert_changes" in result.stdout

        # Verify files were actually created
        desktop = Path.home() / "Desktop"
        plan_files = list(desktop.glob("file_organization_plan_*.md"))
        apply_files = list(desktop.glob("apply_changes_*.py"))
        revert_files = list(desktop.glob("revert_changes_*.py"))

        assert len(plan_files) >= 1, "Plan file should be created"
        assert len(apply_files) >= 1, "Apply script should be created"
        assert len(revert_files) >= 1, "Revert script should be created"

        cleanup_output_files()

    def test_dry_run_no_recurse_skips_subdirs(self, documents_dir, project_root, test_config_file):
        """Test that --no-recurse doesn't traverse subdirectories."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "50",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        assert result.returncode == 0
        # Files should only be from the top level, not nested paths
        # This is hard to verify without knowing the directory structure,
        # but we can at least confirm the command completed

        cleanup_output_files()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_nonexistent_directory(self, project_root, test_config_file):
        """Test handling of non-existent directory."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                "/nonexistent/path/that/does/not/exist",
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        assert result.returncode != 0 or "Not a valid directory" in result.stdout

    def test_missing_config_file(self, project_root):
        """Test handling of missing config file."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(Path.home() / "Documents"),
                "--config",
                "/nonexistent/config.yaml",
                "--dry-run",
                "--auto-continue",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        # Should fail because config file doesn't exist
        assert result.returncode != 0 or "No configuration file found" in result.stdout

    def test_invalid_regex_pattern(self, documents_dir, project_root, test_config_file):
        """Test handling of invalid regex pattern."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
                "--no-recurse",
                "--include",
                "[invalid(regex",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        # Should fail or handle gracefully
        assert result.returncode != 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()


# =============================================================================
# Output Validation Tests
# =============================================================================


class TestOutputValidation:
    """Test that output files are valid."""

    def test_apply_script_is_valid_python(self, documents_dir, project_root, test_config_file):
        """Test that generated apply script is valid Python."""
        cleanup_output_files()

        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "3",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        assert result.returncode == 0

        # Find and validate the apply script
        desktop = Path.home() / "Desktop"
        apply_files = list(desktop.glob("apply_changes_*.py"))

        if apply_files:
            apply_script = apply_files[0]
            # Check it's valid Python by compiling it
            compile_result = subprocess.run(
                ["python", "-m", "py_compile", str(apply_script)],
                capture_output=True,
                text=True,
            )
            assert compile_result.returncode == 0, f"Apply script has syntax errors: {compile_result.stderr}"

        cleanup_output_files()

    def test_revert_script_is_valid_python(self, documents_dir, project_root, test_config_file):
        """Test that generated revert script is valid Python."""
        cleanup_output_files()

        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "3",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        assert result.returncode == 0

        # Find and validate the revert script
        desktop = Path.home() / "Desktop"
        revert_files = list(desktop.glob("revert_changes_*.py"))

        if revert_files:
            revert_script = revert_files[0]
            # Check it's valid Python by compiling it
            compile_result = subprocess.run(
                ["python", "-m", "py_compile", str(revert_script)],
                capture_output=True,
                text=True,
            )
            assert compile_result.returncode == 0, f"Revert script has syntax errors: {compile_result.stderr}"

        cleanup_output_files()

    def test_plan_file_is_markdown(self, documents_dir, project_root, test_config_file):
        """Test that generated plan file is valid markdown."""
        cleanup_output_files()

        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--dry-run",
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "3",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

        assert result.returncode == 0

        # Find and check the plan file
        desktop = Path.home() / "Desktop"
        plan_files = list(desktop.glob("file_organization_plan_*.md"))

        if plan_files:
            plan_file = plan_files[0]
            content = plan_file.read_text()
            # Dry run produces a simple placeholder
            assert len(content) > 0, "Plan file should not be empty"

        cleanup_output_files()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
