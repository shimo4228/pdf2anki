# Contributing to pdf2anki

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Anthropic API key

## Environment Setup

```bash
# Clone repository
git clone https://github.com/shimo4228/Anki-QA.git
cd Anki-QA

# Install all dependencies (including dev + OCR extras)
uv sync --all-extras

# Copy environment variables
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Claude API key for card generation |
| `PDF2ANKI_MODEL` | No | `claude-sonnet-4-5-20250929` | Override the Claude model |
| `PDF2ANKI_BUDGET_LIMIT` | No | `1.00` | API cost budget limit in USD |

## Project Structure

```
src/pdf2anki/
  __init__.py       # Package metadata (version)
  main.py           # CLI entry point (typer commands: convert, preview)
  config.py         # YAML + env var configuration loader
  schemas.py        # Pydantic models (AnkiCard, ExtractionResult, etc.)
  extract.py        # Text extraction (pymupdf4llm + OCR fallback)
  structure.py      # LLM structured card extraction via Claude API
  prompts.py        # Wozniak-based prompt templates
  quality.py        # Quality assurance pipeline (confidence + critique)
  convert.py        # TSV/JSON output conversion
  cost.py           # API cost tracking
tests/
  conftest.py       # Shared fixtures
  fixtures/         # Sample input files (sample.md, sample.txt)
  test_*.py         # Unit tests for each module
```

## CLI Commands

### `pdf2anki convert`

Convert PDF/TXT/MD to Anki flashcards.

```bash
pdf2anki convert input.pdf                     # Basic (TSV output, basic QA)
pdf2anki convert input.pdf -o output.tsv       # Specify output path
pdf2anki convert input.pdf --format json       # JSON output
pdf2anki convert input.pdf --format both       # TSV + JSON
pdf2anki convert input.pdf --quality full      # Full QA pipeline
pdf2anki convert input.pdf --quality off       # Skip QA
pdf2anki convert ./docs/                       # Process entire directory
pdf2anki convert input.pdf --ocr --lang jpn+eng  # Enable OCR
pdf2anki convert input.pdf --max-cards 20      # Limit cards
pdf2anki convert input.pdf --tags "chapter1,important"  # Add tags
pdf2anki convert input.pdf --focus "machine learning"   # Focus topics
pdf2anki convert input.pdf --budget-limit 0.50 # Set budget
pdf2anki convert input.pdf --verbose           # Debug logging
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `input_path` | Argument | required | Input file or directory (PDF/TXT/MD) |
| `-o, --output` | str | auto | Output file or directory |
| `--format` | enum | `tsv` | Output format: `tsv`, `json`, `both` |
| `--quality` | enum | `basic` | QA level: `off`, `basic`, `full` |
| `--model` | str | from config | Claude model name |
| `--max-cards` | int | `50` | Maximum cards to generate |
| `--tags` | str | - | Additional tags (comma-separated) |
| `--focus` | str | - | Focus topics (comma-separated) |
| `--card-types` | str | all 7 | Card types to generate (comma-separated) |
| `--bloom-filter` | str | all | Bloom levels to include (comma-separated) |
| `--budget-limit` | float | `1.00` | Budget limit in USD |
| `--ocr` | flag | off | Enable OCR for image-heavy PDFs |
| `--lang` | str | `jpn+eng` | OCR language |
| `--config` | str | `config.yaml` | Path to config YAML file |
| `--verbose` | flag | off | Enable debug logging |

### `pdf2anki preview`

Dry-run text extraction (no API calls).

```bash
pdf2anki preview input.pdf
pdf2anki preview input.pdf --ocr
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `input_path` | Argument | required | Input file (PDF/TXT/MD) |
| `--ocr` | flag | off | Enable OCR |
| `--lang` | str | `jpn+eng` | OCR language |
| `--verbose` | flag | off | Enable debug logging |

## Configuration

Configuration priority: **env vars > config.yaml > defaults**.

See [config.yaml](../config.yaml) for all available settings including:
- Claude API model and token limits
- Quality pipeline thresholds and critique rounds
- Card generation limits and types
- Cost tracking budget and warnings
- OCR language settings

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=pdf2anki --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_schemas.py

# Run specific test
uv run pytest tests/test_schemas.py::test_anki_card_creation -v

# Run tests matching pattern
uv run pytest -k "test_config" -v
```

**Coverage target: 80%** (enforced via `pyproject.toml`).

Total test count: **266 tests** across 9 test modules.

## Development Workflow

1. **Plan** - Create implementation plan for complex changes
2. **Write tests first** (TDD) - RED phase
3. **Run tests** - Confirm they fail
4. **Implement** - GREEN phase (minimal code to pass)
5. **Refactor** - IMPROVE phase
6. **Verify coverage** - Must be 80%+
7. **Commit** - Use conventional commits (`feat:`, `fix:`, `refactor:`, etc.)

## Supported File Types

| Extension | Method | Notes |
|-----------|--------|-------|
| `.pdf` | pymupdf4llm | Optional OCR fallback via ocrmypdf |
| `.txt` | Plain text read | UTF-8 |
| `.md` | Plain text read | UTF-8 |

## Card Types

| Type | Description |
|------|-------------|
| `qa` | Question and answer |
| `term_definition` | Term with definition |
| `summary_point` | Key summary point |
| `cloze` | Fill-in-the-blank (Anki cloze deletion) |
| `reversible` | Bidirectional card (generates 2 rows in TSV) |
| `sequence` | Ordered steps or processes |
| `compare_contrast` | Comparison between concepts |
| `image_occlusion` | Image-based (schema only) |

## Bloom's Taxonomy Levels

Every card is tagged with a cognitive level: `remember`, `understand`, `apply`, `analyze`, `evaluate`, `create`.
