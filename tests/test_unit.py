#!/usr/bin/env python3
"""
Unit tests for the renamer application.

Tests core logic without external dependencies (no LLM, no GUI).

Run with: uv run pytest tests/test_unit.py -v
"""

import re
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from renamer import (
    AnalysisResult,
    Config,
    ConfigurationProfile,
    FileRecord,
    ModelDefinition,
    RenamerConfig,
    TempFileManager,
    expand_env_vars,
    generate_example_config,
)

# =============================================================================
# Config Tests
# =============================================================================


class TestConfig:
    """Tests for the Config dataclass (runtime non-LLM settings)."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        assert config.max_content_chars == 50000
        assert config.max_pages == 3

    def test_supported_extensions(self):
        """Test that all expected extensions are supported."""
        config = Config()
        expected = {
            ".pdf",
            ".md",
            ".txt",
            ".jpg",
            ".jpeg",
            ".png",
            ".docx",
            ".xlsx",
            ".key",
            ".numbers",
            ".pages",
        }

        assert config.supported_extensions == expected

    def test_prompts_dir_exists(self):
        """Test that prompts directory path is set correctly."""
        config = Config()
        # Should be relative to the script location
        assert "prompts" in str(config.prompts_dir)


class TestModelDefinition:
    """Tests for the ModelDefinition Pydantic model."""

    def test_create_ollama_model(self):
        """Test creating an Ollama model definition."""
        model = ModelDefinition(
            name="llava-local",
            provider="ollama",
            model="llava:13b",
            endpoint="http://localhost:11434",
        )

        assert model.name == "llava-local"
        assert model.provider == "ollama"
        assert model.model == "llava:13b"
        assert model.endpoint == "http://localhost:11434"
        assert model.api_key is None

    def test_create_anthropic_model(self):
        """Test creating an Anthropic model definition."""
        model = ModelDefinition(
            name="claude-sonnet",
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key="test-api-key",
        )

        assert model.name == "claude-sonnet"
        assert model.provider == "anthropic"
        assert model.api_key == "test-api-key"

    def test_create_google_model(self):
        """Test creating a Google model definition."""
        model = ModelDefinition(
            name="gemini-flash",
            provider="google",
            model="gemini-2.0-flash",
            api_key="test-google-key",
        )

        assert model.provider == "google"
        assert model.model == "gemini-2.0-flash"


class TestConfigurationProfile:
    """Tests for the ConfigurationProfile Pydantic model."""

    def test_create_profile(self):
        """Test creating a configuration profile."""
        profile = ConfigurationProfile(
            name="local-only",
            description="All local models",
            ocr_model="llava-local",
            analysis_model="llama32-local",
            organizer_model="gemma3-local",
        )

        assert profile.name == "local-only"
        assert profile.ocr_model == "llava-local"
        assert profile.analysis_model == "llama32-local"
        assert profile.organizer_model == "gemma3-local"

    def test_profile_default_description(self):
        """Test that description defaults to empty string."""
        profile = ConfigurationProfile(
            name="test",
            ocr_model="model1",
            analysis_model="model2",
            organizer_model="model3",
        )

        assert profile.description == ""


class TestRenamerConfig:
    """Tests for the RenamerConfig Pydantic model."""

    def test_create_config(self):
        """Test creating a full renamer configuration."""
        config = RenamerConfig(
            models=[
                ModelDefinition(name="model1", provider="ollama", model="llama3.2"),
                ModelDefinition(name="model2", provider="ollama", model="llava:13b"),
            ],
            configurations=[
                ConfigurationProfile(
                    name="test-profile",
                    ocr_model="model2",
                    analysis_model="model1",
                    organizer_model="model1",
                ),
            ],
            default_configuration="test-profile",
        )

        assert len(config.models) == 2
        assert len(config.configurations) == 1
        assert config.default_configuration == "test-profile"

    def test_get_model(self):
        """Test getting a model by name."""
        config = RenamerConfig(
            models=[
                ModelDefinition(name="model1", provider="ollama", model="llama3.2"),
                ModelDefinition(name="model2", provider="anthropic", model="claude-sonnet"),
            ],
            configurations=[
                ConfigurationProfile(
                    name="test",
                    ocr_model="model1",
                    analysis_model="model2",
                    organizer_model="model1",
                ),
            ],
        )

        model = config.get_model("model2")
        assert model.provider == "anthropic"

    def test_get_model_not_found(self):
        """Test getting a non-existent model raises ValueError."""
        config = RenamerConfig(
            models=[ModelDefinition(name="model1", provider="ollama", model="test")],
            configurations=[
                ConfigurationProfile(
                    name="test",
                    ocr_model="model1",
                    analysis_model="model1",
                    organizer_model="model1",
                ),
            ],
        )

        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            config.get_model("nonexistent")

    def test_get_profile(self):
        """Test getting a profile by name."""
        config = RenamerConfig(
            models=[ModelDefinition(name="model1", provider="ollama", model="test")],
            configurations=[
                ConfigurationProfile(
                    name="profile1",
                    ocr_model="model1",
                    analysis_model="model1",
                    organizer_model="model1",
                ),
                ConfigurationProfile(
                    name="profile2",
                    ocr_model="model1",
                    analysis_model="model1",
                    organizer_model="model1",
                ),
            ],
        )

        profile = config.get_profile("profile2")
        assert profile.name == "profile2"


class TestEnvVarExpansion:
    """Tests for environment variable expansion."""

    def test_expand_single_var(self):
        """Test expanding a single environment variable."""
        import os
        os.environ["TEST_VAR"] = "test_value"

        result = expand_env_vars("prefix_${TEST_VAR}_suffix")
        assert result == "prefix_test_value_suffix"

        del os.environ["TEST_VAR"]

    def test_expand_multiple_vars(self):
        """Test expanding multiple environment variables."""
        import os
        os.environ["VAR1"] = "value1"
        os.environ["VAR2"] = "value2"

        result = expand_env_vars("${VAR1}_and_${VAR2}")
        assert result == "value1_and_value2"

        del os.environ["VAR1"]
        del os.environ["VAR2"]

    def test_missing_var_raises_error(self):
        """Test that missing environment variable raises ValueError."""
        with pytest.raises(ValueError, match="Environment variable 'NONEXISTENT_VAR' is not set"):
            expand_env_vars("${NONEXISTENT_VAR}")

    def test_no_vars_returns_unchanged(self):
        """Test that string without vars is returned unchanged."""
        result = expand_env_vars("no_variables_here")
        assert result == "no_variables_here"


class TestGenerateExampleConfig:
    """Tests for the example config generator."""

    def test_generates_valid_yaml(self):
        """Test that generated config is valid YAML."""
        import yaml

        config_str = generate_example_config()
        config = yaml.safe_load(config_str)

        assert "models" in config
        assert "configurations" in config
        assert "default_configuration" in config

    def test_has_all_providers(self):
        """Test that example config includes all providers."""
        config_str = generate_example_config()

        assert "provider: ollama" in config_str
        assert "provider: anthropic" in config_str
        assert "provider: openai" in config_str
        assert "provider: google" in config_str
        assert "provider: xai" in config_str


# =============================================================================
# FileRecord Tests
# =============================================================================


class TestFileRecord:
    """Tests for the FileRecord dataclass."""

    def test_create_file_record(self):
        """Test creating a FileRecord."""
        record = FileRecord(
            path=Path("/test/file.pdf"),
            size=1024,
            modified=datetime(2024, 1, 15, 10, 30, 0),
            extension=".pdf",
        )

        assert record.path == Path("/test/file.pdf")
        assert record.size == 1024
        assert record.extension == ".pdf"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        record = FileRecord(
            path=Path("/test/file.pdf"),
            size=1024,
            modified=datetime(2024, 1, 15, 10, 30, 0),
            extension=".pdf",
        )

        d = record.to_dict()

        assert d["path"] == "/test/file.pdf"
        assert d["size"] == 1024
        assert d["extension"] == ".pdf"
        assert "2024-01-15" in d["modified"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "path": "/test/file.pdf",
            "size": 2048,
            "modified": "2024-03-20T14:00:00",
            "extension": ".pdf",
        }

        record = FileRecord.from_dict(d)

        assert record.path == Path("/test/file.pdf")
        assert record.size == 2048
        assert record.extension == ".pdf"
        assert record.modified.year == 2024
        assert record.modified.month == 3

    def test_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        original = FileRecord(
            path=Path("/test/document.docx"),
            size=5000,
            modified=datetime(2024, 6, 1, 9, 15, 30),
            extension=".docx",
        )

        restored = FileRecord.from_dict(original.to_dict())

        assert restored.path == original.path
        assert restored.size == original.size
        assert restored.extension == original.extension


# =============================================================================
# AnalysisResult Tests
# =============================================================================


class TestAnalysisResult:
    """Tests for the AnalysisResult dataclass."""

    def test_create_analysis_result(self):
        """Test creating an AnalysisResult."""
        result = AnalysisResult(
            original_path="/test/invoice.pdf",
            original_name="invoice.pdf",
            proposed_name="2024-01-15-acme-invoice",
            tags=["invoice", "finance"],
            summary="Invoice from Acme Corp.",
            analysis_model="llama3.2",
            analysis_time="00:02.500",
            analysis_tokens_in=1500,
            analysis_tokens_out=100,
        )

        assert result.original_name == "invoice.pdf"
        assert result.proposed_name == "2024-01-15-acme-invoice"
        assert "invoice" in result.tags
        assert result.error is None

    def test_analysis_result_with_error(self):
        """Test AnalysisResult with an error."""
        result = AnalysisResult(
            original_path="/test/corrupt.pdf",
            original_name="corrupt.pdf",
            error="File is corrupted",
        )

        assert result.error == "File is corrupted"
        assert result.proposed_name == ""
        assert result.tags == []

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = AnalysisResult(
            original_path="/test/doc.pdf",
            original_name="doc.pdf",
            proposed_name="new-name",
            tags=["tag1", "tag2"],
            summary="A document.",
            analysis_model="llama3.2",
        )

        d = result.to_dict()

        assert d["original_path"] == "/test/doc.pdf"
        assert d["proposed_name"] == "new-name"
        assert d["tags"] == ["tag1", "tag2"]
        assert d["error"] is None

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "original_path": "/test/file.pdf",
            "original_name": "file.pdf",
            "ocr_model": "llava:13b",
            "ocr_time": "00:05.000",
            "ocr_tokens_in": 500,
            "ocr_tokens_out": 200,
            "proposed_name": "analyzed-file",
            "tags": ["document"],
            "summary": "A file.",
            "analysis_model": "llama3.2",
            "analysis_time": "00:01.000",
            "analysis_tokens_in": 1000,
            "analysis_tokens_out": 50,
            "error": None,
        }

        result = AnalysisResult.from_dict(d)

        assert result.ocr_model == "llava:13b"
        assert result.proposed_name == "analyzed-file"

    def test_ocr_fields_default_to_zero(self):
        """Test that OCR fields default correctly."""
        result = AnalysisResult(
            original_path="/test/text.txt",
            original_name="text.txt",
        )

        assert result.ocr_model == ""
        assert result.ocr_time == "00:00.000"
        assert result.ocr_tokens_in == 0
        assert result.ocr_tokens_out == 0


# =============================================================================
# TempFileManager Tests
# =============================================================================


class TestTempFileManager:
    """Tests for the TempFileManager class."""

    def test_initialize_creates_directory(self):
        """Test that initialize creates the session directory."""
        config = Config()
        manager = TempFileManager(config)

        session_dir = manager.initialize()

        assert session_dir.exists()
        assert session_dir.is_dir()
        assert "session_" in session_dir.name

        # Cleanup
        session_dir.rmdir()

    def test_get_temp_path(self):
        """Test getting a temp file path."""
        config = Config()
        manager = TempFileManager(config)
        manager.initialize()

        path = manager.get_temp_path("test_file.jsonl")

        assert path.name == "test_file.jsonl"
        assert path.parent == manager.session_dir

        # Cleanup
        if manager.session_dir:
            manager.session_dir.rmdir()

    def test_append_and_read_jsonl(self):
        """Test writing and reading JSONL files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Write records
            records = [
                {"id": 1, "name": "first"},
                {"id": 2, "name": "second"},
                {"id": 3, "name": "third"},
            ]

            for record in records:
                TempFileManager.append_jsonl(temp_path, record)

            # Read back
            read_records = list(TempFileManager.read_jsonl(temp_path))

            assert len(read_records) == 3
            assert read_records[0]["id"] == 1
            assert read_records[1]["name"] == "second"
            assert read_records[2]["id"] == 3

        finally:
            temp_path.unlink()

    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        records = list(TempFileManager.read_jsonl(Path("/nonexistent/file.jsonl")))
        assert records == []

    def test_jsonl_unicode_handling(self):
        """Test that JSONL handles unicode correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            record = {"name": "日本語テスト", "emoji": "📄"}
            TempFileManager.append_jsonl(temp_path, record)

            read_records = list(TempFileManager.read_jsonl(temp_path))

            assert read_records[0]["name"] == "日本語テスト"
            assert read_records[0]["emoji"] == "📄"

        finally:
            temp_path.unlink()


# =============================================================================
# Pattern Matching Tests
# =============================================================================


class TestPatternMatching:
    """Tests for regex pattern matching logic."""

    def test_include_pattern_matches(self):
        """Test that include pattern correctly matches files."""
        pattern = re.compile(r".*\.pdf$")

        assert pattern.search("invoice.pdf")
        assert pattern.search("document.PDF") is None  # Case sensitive
        assert pattern.search("file.txt") is None

    def test_include_pattern_all_files(self):
        """Test the default include pattern matches everything."""
        pattern = re.compile(r".*")

        assert pattern.search("any_file.pdf")
        assert pattern.search("another.txt")
        assert pattern.search("")

    def test_exclude_pattern(self):
        """Test that exclude pattern correctly excludes files."""
        pattern = re.compile(r"^temp_|\.bak$")

        assert pattern.search("temp_file.pdf")
        assert pattern.search("document.bak")
        assert pattern.search("normal.pdf") is None

    def test_combined_patterns(self):
        """Test include and exclude patterns working together."""
        include = re.compile(r".*\.pdf$")
        exclude = re.compile(r"^draft_")

        test_files = [
            ("invoice.pdf", True),  # Included, not excluded
            ("draft_report.pdf", False),  # Included but excluded
            ("notes.txt", False),  # Not included
            ("draft_notes.txt", False),  # Not included
        ]

        for filename, should_include in test_files:
            matches_include = bool(include.search(filename))
            matches_exclude = bool(exclude.search(filename))
            result = matches_include and not matches_exclude

            assert result == should_include, f"Failed for {filename}"


# =============================================================================
# Filename Handling Tests
# =============================================================================


class TestFilenameHandling:
    """Tests for filename sanitization and handling."""

    def test_hidden_file_detection(self):
        """Test detection of hidden files."""
        hidden_files = [".hidden", ".DS_Store", ".gitignore"]
        normal_files = ["file.txt", "document.pdf", "image.png"]

        for f in hidden_files:
            assert f.startswith(".")

        for f in normal_files:
            assert not f.startswith(".")

    def test_extension_extraction(self):
        """Test extracting file extensions."""
        test_cases = [
            ("file.pdf", ".pdf"),
            ("document.DOCX", ".DOCX"),
            ("image.tar.gz", ".gz"),
            ("no_extension", ""),
            (".hidden", ""),
        ]

        for filename, expected_ext in test_cases:
            path = Path(filename)
            assert path.suffix == expected_ext

    def test_extension_normalization(self):
        """Test normalizing extensions to lowercase."""
        extensions = [".PDF", ".DOCX", ".Jpg", ".PNG"]

        for ext in extensions:
            normalized = ext.lower()
            assert normalized == normalized.lower()
            assert normalized.startswith(".")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_analysis_result(self):
        """Test AnalysisResult with minimal data."""
        result = AnalysisResult(
            original_path="",
            original_name="",
        )

        d = result.to_dict()
        restored = AnalysisResult.from_dict(d)

        assert restored.original_path == ""
        assert restored.tags == []

    def test_very_long_filename(self):
        """Test handling of very long filenames."""
        long_name = "a" * 300 + ".pdf"
        path = Path(long_name)

        # Path object handles long names
        assert len(path.name) == 304
        assert path.suffix == ".pdf"

        # Truncation logic (if implemented)
        max_len = 255
        if len(path.stem) > max_len - len(path.suffix):
            truncated = path.stem[: max_len - len(path.suffix)] + path.suffix
            assert len(truncated) <= max_len

    def test_special_characters_in_filename(self):
        """Test filenames with special characters."""
        special_names = [
            "file with spaces.pdf",
            "file-with-dashes.pdf",
            "file_with_underscores.pdf",
            "file.multiple.dots.pdf",
            "résumé.pdf",
            "日本語ファイル.pdf",
        ]

        for name in special_names:
            path = Path(name)
            assert path.name == name
            assert path.suffix == ".pdf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
