# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
uv run renamer.py [OPTIONS] [DIRECTORY]
```

If no directory is provided, a native macOS file picker opens.

## Testing

```bash
# Run all tests
uv run pytest

# Run unit tests only (no LLM or GUI dependencies)
uv run pytest tests/test_unit.py -v

# Run a single test
uv run pytest tests/test_unit.py::TestConfig::test_default_config -v
```

## Architecture

This is a single-file Python application (`renamer.py`). Dependencies are declared once, in `pyproject.toml`, and pinned by `uv.lock` — `renamer.py` deliberately carries no PEP 723 inline metadata block, so `uv run renamer.py` executes against the locked project environment rather than an unlocked ad-hoc one.

Note that `renamer.py` *generates* standalone apply/revert scripts that do carry their own PEP 723 blocks. Their `# /// script` markers are assembled from the `PEP723_OPEN`/`PEP723_CLOSE` constants instead of being written literally, because uv scans a file's raw text for those markers — a literal one at column 0 inside the generator templates would make uv see multiple metadata blocks in `renamer.py` and refuse to run it.

It processes files in three sequential passes:

1. **Discovery Pass** (`DiscoveryPass`): Enumerates files matching include/exclude patterns, writes to `pass1_discovery.jsonl`
2. **Analysis Pass** (`AnalysisPass`): Extracts content, performs OCR if needed, gets LLM-suggested names/tags/summaries, writes to `pass2_analysis.jsonl`
3. **Organization Pass** (`OrganizationPass`): Generates folder structure proposal and Python scripts to apply/revert changes

### Key Components

- **LLMManager**: Multi-provider LLM support via LangChain (Ollama, Anthropic, OpenAI, Google, xAI). Configured via `renamer.yaml`.
- **ContentExtractor**: Extracts text/images from PDFs, DOCX, XLSX, images, and Apple iWork formats (via AppleScript export)
- **MacOSDialogs**: Native macOS dialogs using PyObjC (NSOpenPanel, NSAlert)
- **TempFileManager**: Session-based temp files with Trash cleanup

### Configuration

LLM providers and model assignments are configured in `renamer.yaml` (searched in cwd, then `~/.config/renamer/config.yaml`). Configuration profiles assign models to roles: `ocr_model`, `analysis_model`, `organizer_model`.

Optional `family.yaml` provides context about family members for document analysis.

### Prompts

LLM prompts are stored in `prompts/`:
- `ocr_vision.txt`: Vision model OCR instructions
- `document_analysis.txt`: Name/tags/summary extraction
- `organization_planning.txt`: Folder structure generation

## Git Conventions

- Use conventional commit messages (e.g., `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`)
- Do not run git status or git diff before committing - just add and commit directly
