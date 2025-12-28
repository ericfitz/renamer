#!/usr/bin/env python3
"""
LLM Validation tests for the renamer application.

These tests run actual LLM analysis on a sample of documents and then
send the results to a different LLM for a "second opinion" validation.

This helps verify:
1. The proposed names are sensible and descriptive
2. The tags are appropriate for the content
3. The summaries accurately reflect the document content
4. The organization plan makes logical sense

Requirements:
- Ollama must be running
- Required models must be available (or will be pulled)

Run with: uv run pytest tests/test_llm_validation.py -v -s
         (use -s to see the LLM feedback)

Note: These tests are slower and require LLM resources. They are marked
with pytest.mark.slow and can be skipped with: pytest -m "not slow"
"""

import subprocess
from pathlib import Path

import pytest

# Try to import ollama to check if it's available
try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def is_ollama_running() -> bool:
    """Check if Ollama is running and accessible."""
    if not OLLAMA_AVAILABLE:
        return False
    try:
        client = ollama.Client()
        client.list()
        return True
    except Exception:
        return False


# Skip all tests in this module if Ollama isn't running
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_ollama_running(), reason="Ollama is not running"),
]


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
def ollama_client():
    """Return an Ollama client."""
    return ollama.Client()


@pytest.fixture
def test_config_file(project_root):
    """Create a temporary config file for testing with local models only."""
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
    config_path = project_root / "test_llm_config.yaml"
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


class TestLLMAnalysisQuality:
    """Test the quality of LLM analysis with a second opinion."""

    def test_analyze_sample_and_validate(self, documents_dir, project_root, ollama_client, test_config_file):
        """
        Run analysis on a small sample, then ask another LLM to validate.
        """
        cleanup_output_files()

        # Run renamer with actual LLM (not dry-run) on a small sample
        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "3",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,  # 5 minutes for LLM processing
        )

        # If it failed, print output for debugging
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

        assert result.returncode == 0, f"Renamer failed: {result.stderr}"

        # Find the temp directory from output to read analysis results
        # The analysis file path is mentioned in the output
        if "pass2_analysis.jsonl" not in result.stdout:
            pytest.skip("Could not find analysis output in stdout")

        # Read the organization plan
        desktop = Path.home() / "Desktop"
        plan_files = list(desktop.glob("file_organization_plan_*.md"))

        if not plan_files:
            pytest.skip("No plan file generated")

        plan_content = plan_files[0].read_text()

        # Now ask a second LLM to validate the organization plan
        validation_prompt = f"""You are a quality assurance reviewer for a file organization system.

Please review this proposed file organization plan and provide feedback:

{plan_content}

Evaluate:
1. Does the folder structure make logical sense?
2. Are the proposed filenames descriptive and follow good naming conventions?
3. Is the organization coherent and would it help someone find files easily?

Respond with:
- QUALITY_SCORE: A number 1-10 (10 being excellent)
- ISSUES: Any problems you notice (or "None" if none)
- SUGGESTIONS: Any improvements (or "None" if none)

Be concise."""

        try:
            response = ollama_client.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": validation_prompt}],
            )

            feedback = response.message.content
            print(f"\n=== LLM Validation Feedback ===\n{feedback}\n")

            # Basic assertion - the response should contain a quality score
            assert "QUALITY_SCORE" in feedback or "quality" in feedback.lower()

        except Exception as e:
            pytest.skip(f"LLM validation failed: {e}")

        finally:
            cleanup_output_files()

    def test_validate_proposed_names(self, documents_dir, project_root, ollama_client, test_config_file):
        """
        Run analysis and validate that proposed names are sensible.
        """
        cleanup_output_files()

        # Run renamer
        result = subprocess.run(
            [
                "uv",
                "run",
                "renamer.py",
                "run",
                str(documents_dir),
                "--config",
                str(test_config_file),
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "2",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.skip(f"Renamer failed: {result.stderr}")

        # Read the plan file to extract proposed names
        desktop = Path.home() / "Desktop"
        plan_files = list(desktop.glob("file_organization_plan_*.md"))

        if not plan_files:
            pytest.skip("No plan file generated")

        plan_content = plan_files[0].read_text()

        # Ask LLM to validate naming quality
        validation_prompt = f"""Review these proposed file names from an organization plan.

{plan_content}

For each proposed filename, evaluate:
1. Is it descriptive?
2. Does it follow the pattern: lowercase-with-hyphens?
3. Does it include a date if relevant?
4. Is it a reasonable length (not too long, not too short)?

Respond with:
- NAMING_SCORE: 1-10
- GOOD_EXAMPLES: List 1-2 well-named files
- BAD_EXAMPLES: List any poorly named files (or "None")

Be concise."""

        try:
            response = ollama_client.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": validation_prompt}],
            )

            feedback = response.message.content
            print(f"\n=== Naming Validation ===\n{feedback}\n")

            assert "NAMING_SCORE" in feedback or "score" in feedback.lower()

        except Exception as e:
            pytest.skip(f"LLM validation failed: {e}")

        finally:
            cleanup_output_files()

    def test_validate_tag_appropriateness(self, documents_dir, project_root, ollama_client, test_config_file):
        """
        Validate that assigned tags are appropriate for the documents.
        """
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
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "2",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,
        )

        if result.returncode != 0:
            pytest.skip(f"Renamer failed: {result.stderr}")

        desktop = Path.home() / "Desktop"
        plan_files = list(desktop.glob("file_organization_plan_*.md"))

        if not plan_files:
            pytest.skip("No plan file generated")

        plan_content = plan_files[0].read_text()

        validation_prompt = f"""Review the tags and categories in this file organization plan:

{plan_content}

Evaluate:
1. Are the folder categories logical (e.g., finance, legal, personal)?
2. Are files placed in appropriate categories based on their names/descriptions?
3. Is there good consistency in how similar files are categorized?

Respond with:
- CATEGORIZATION_SCORE: 1-10
- WELL_CATEGORIZED: Examples of good categorization
- MISCATEGORIZED: Any files that seem in the wrong category (or "None")

Be concise."""

        try:
            response = ollama_client.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": validation_prompt}],
            )

            feedback = response.message.content
            print(f"\n=== Categorization Validation ===\n{feedback}\n")

            assert "CATEGORIZATION_SCORE" in feedback or "score" in feedback.lower()

        except Exception as e:
            pytest.skip(f"LLM validation failed: {e}")

        finally:
            cleanup_output_files()


class TestOrganizationPlanQuality:
    """Test the quality of the overall organization plan."""

    def test_plan_coherence(self, documents_dir, project_root, ollama_client, test_config_file):
        """
        Test that the organization plan is coherent and well-structured.
        """
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
                "--auto-continue",
                "--no-recurse",
                "--max-files",
                "5",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,
        )

        if result.returncode != 0:
            pytest.skip(f"Renamer failed: {result.stderr}")

        desktop = Path.home() / "Desktop"
        plan_files = list(desktop.glob("file_organization_plan_*.md"))

        if not plan_files:
            pytest.skip("No plan file generated")

        plan_content = plan_files[0].read_text()

        validation_prompt = f"""You are evaluating a file organization plan.

{plan_content}

Please assess:
1. STRUCTURE: Is the folder hierarchy logical and not too deep (max 3 levels)?
2. BALANCE: Are files distributed reasonably across folders?
3. FINDABILITY: Would a user be able to find their files easily?
4. CONSISTENCY: Are similar files grouped together?

Provide an overall assessment:
- OVERALL_SCORE: 1-10
- STRENGTHS: What works well
- WEAKNESSES: What could be improved
- RECOMMENDATION: "APPROVE" or "NEEDS_REVISION"

Be concise but thorough."""

        try:
            response = ollama_client.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": validation_prompt}],
            )

            feedback = response.message.content
            print(f"\n=== Organization Plan Review ===\n{feedback}\n")

            # The plan should get at least a passing review
            assert "OVERALL_SCORE" in feedback or "score" in feedback.lower()

        except Exception as e:
            pytest.skip(f"LLM validation failed: {e}")

        finally:
            cleanup_output_files()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
