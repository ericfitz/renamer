# File Renamer & Organizer - Implementation Plan

## Project Overview
A macOS Python application to rename and organize local files using LLM-powered analysis. The application works in three passes:
1. **Discovery Pass**: Enumerate files matching include/exclude regex patterns
2. **Analysis Pass**: Extract metadata, OCR if needed, get LLM-suggested names/tags/summaries
3. **Organization Pass**: Propose folder structure, generate rename/move scripts

## Confirmed Architectural Decisions

| Decision | Choice |
|----------|--------|
| GUI dialogs | PyObjC (native macOS NSOpenPanel, NSAlert) |
| CLI framework | typer |
| LLM library | LangChain with Ollama |
| Apple formats | AppleScript export to PDF |
| Legacy .doc | Skip (only support .docx) |
| Temp file format | JSONL |
| Error handling | Skip and log problematic files |
| Temp cleanup | Move to macOS Trash |
| Output scripts | Python |
| Prompt storage | `prompts/` subfolder |
| Model auto-pull | Yes, auto-pull if missing |
| Dry run mode | Yes, `--dry-run` flag |
| Hidden files | Exclude by default |
| File operation | Move (not copy) |
| OCR behavior | Modify original PDF in place (copy if read-only) |
| Output location | Within original directory |
| Resumability | Yes, support --resume flag |

---

## Project Structure

```
renamer/
├── renamer.py                      # Main entry point (single-file with inline deps)
├── prompts/
│   ├── ocr_vision.txt              # Prompt for image/PDF OCR via vision model
│   ├── document_analysis.txt       # Prompt for extracting name/tags/summary
│   └── organization_planning.txt   # Prompt for folder structure generation
├── PLAN.md                         # This file
├── README.md
├── .gitignore
└── LICENSE
```

---

## Dependencies (Inline uv Script Header)

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "typer>=0.15.0",
#   "rich>=13.0.0",
#   "pyobjc-framework-Cocoa>=10.0",
#   "langchain-ollama>=0.2.0",
#   "langchain-core>=0.3.0",
#   "pymupdf>=1.25.0",
#   "python-docx>=1.1.0",
#   "openpyxl>=3.1.0",
#   "pillow>=11.0.0",
#   "send2trash>=1.8.0",
#   "ollama>=0.4.0",
# ]
# ///
```

---

## CLI Interface

```bash
uv run renamer.py [OPTIONS] [DIRECTORY]

Options:
  --recurse / --no-recurse    Process subdirectories (default: True)
  -i, --include TEXT          Include regex pattern (default: ".*")
  -e, --exclude TEXT          Exclude regex pattern (default: "")
  -n, --max-files INTEGER     Maximum files to process
  --dry-run                   Skip LLM calls, show what would be processed
  --auto-continue             Skip confirmation prompts between passes
  --resume                    Resume from previous interrupted run
  --vision-model TEXT         Ollama vision model (default: "llava:13b")
  --analysis-model TEXT       Ollama analysis model (default: "llama3.2")
```

If `DIRECTORY` is not provided, show native macOS file picker dialog.

---

## File Type Support

| Format | Extensions | Library | Notes |
|--------|------------|---------|-------|
| PDF | .pdf | PyMuPDF (fitz) | OCR via vision model if no text layer |
| Markdown | .md | Built-in | Direct read |
| Text | .txt | Built-in | Direct read |
| JPEG | .jpg, .jpeg | Pillow | LLM vision for analysis |
| PNG | .png | Pillow | LLM vision for analysis |
| Word | .docx | python-docx | Text extraction |
| Excel | .xlsx | openpyxl | Sheet names + sample data |
| Keynote | .key | AppleScript→PDF | Export then extract |
| Numbers | .numbers | AppleScript→PDF | Export then extract |
| Pages | .pages | AppleScript→PDF | Export then extract |

---

## Ollama Models (64GB RAM)

| Task | Recommended Model | VRAM Usage |
|------|-------------------|------------|
| Vision/OCR | llava:13b | ~8GB |
| Document Analysis | llama3.2 | ~5GB |
| Organization Planning | llama3.2 | ~5GB |

Models auto-pulled if missing via `ollama pull`.

---

## Three-Pass Architecture

### Pass 1: Discovery

1. Get directory (CLI arg or native file picker)
2. Get options (CLI args or native dialog):
   - Recurse subdirectories? (default: yes)
   - Include pattern (default: `.*`)
   - Exclude pattern (default: empty)
   - Max files (default: unlimited)
3. Walk directory tree, apply patterns
4. Write matching files to temp JSONL: `pass1_discovery.jsonl`
5. Display file list to user
6. Prompt to continue (unless `--auto-continue`)

**Discovery JSONL format:**
```json
{"path": "/path/to/file.pdf", "size": 45632, "modified": "2024-12-01T10:30:00", "extension": ".pdf"}
```

### Pass 2: Analysis

For each file in discovery list:
1. Check if already processed (for `--resume` support)
2. Display progress: `[12/50] Processing: invoice.pdf - Extracting text...`
3. Extract content based on file type (first 3 pages max)
4. If image or scanned PDF → use vision model for OCR, write OCR text layer back to PDF (if read-only, create writable copy with `-ocr` suffix)
5. Send content to LLM with `document_analysis.txt` prompt
6. Parse response: proposed name, tags, summary
7. Append to `pass2_analysis.jsonl` with timing/token stats
8. Continue to next file

**Resume support**: The `--resume` flag checks `pass2_analysis.jsonl` for already-processed files and skips them.

**Analysis JSONL format:**
```json
{
  "original_path": "/path/to/invoice.pdf",
  "original_name": "invoice.pdf",
  "ocr_model": "",
  "ocr_time": "00:00.000",
  "ocr_tokens_in": 0,
  "ocr_tokens_out": 0,
  "proposed_name": "2024-12-01-acme-invoice-1234",
  "tags": ["invoice", "finance", "dated"],
  "summary": "Invoice from Acme Corp for consulting services.",
  "analysis_model": "llama3.2",
  "analysis_time": "00:02.340",
  "analysis_tokens_in": 1523,
  "analysis_tokens_out": 89
}
```

### Pass 3: Organization

1. Load all records from `pass2_analysis.jsonl`
2. Send to LLM with `organization_planning.txt` prompt
3. LLM proposes folder structure and file mappings
4. Generate outputs on Desktop:
   - `file_organization_plan.md` - Proposed structure with reasoning
   - `apply_changes.py` - Script to execute the organization (moves files within original directory)
   - `revert_changes.py` - Script to undo all changes
5. Move temp files to Trash

**Note**: Organized files are moved into new subfolders within the original scanned directory.

---

## Prompt Files

### prompts/ocr_vision.txt
Instructs vision model to:
- Extract all visible text from image
- Preserve document structure
- Identify document type, title, dates
- Mark illegible sections

### prompts/document_analysis.txt
Instructs LLM to return JSON with:
- `proposed_name`: Descriptive filename (lowercase-hyphenated, max 60 chars)
- `tags`: 3-7 classification tags
- `summary`: 1-2 sentence description

### prompts/organization_planning.txt
Instructs LLM to:
- Analyze all file summaries and tags
- Propose hierarchical folder structure (max 3 levels deep)
- Map each file to destination folder and new name
- Return markdown structure + JSON move list

---

## Error Handling

| Error Type | Handling |
|------------|----------|
| File access denied | Skip, log to error list |
| Corrupted file | Skip, log to error list |
| Password-protected | Skip, log to error list |
| AppleScript timeout | Skip, log to error list |
| Ollama not running | Exit with clear instructions |
| Model pull failed | Retry once, then exit |
| Malformed LLM response | Retry once with simpler prompt |

All skipped files logged to `errors.log` in temp directory.

---

## Generated Scripts

### apply_changes.py
- Creates folder structure
- Moves/renames files to new locations
- Handles conflicts with numeric suffix
- Logs all operations

### revert_changes.py
- Reads operations from apply script's log
- Moves files back to original locations
- Removes empty folders created by apply
- Uses send2trash for safety

---

## Edge Cases

- **Empty directory**: Exit gracefully with message
- **No matching files**: Warn and exit after pattern filtering
- **Symbolic links**: Skip by default
- **Hidden files**: Exclude by default (files starting with `.`)
- **Very long filenames**: Truncate to filesystem limit (255 chars)
- **Unicode filenames**: Handle with pathlib
- **Files with no extension**: Process based on content
- **Filename conflicts**: Add numeric suffix (-1, -2, etc.)
- **Cross-filesystem moves**: Copy then delete original
- **Read-only PDFs needing OCR**: Create writable copy with `-ocr` suffix
