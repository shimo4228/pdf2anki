# pdf2anki

[日本語版はこちら / Japanese](README.ja.md)

A CLI tool that automatically generates high-quality Anki flashcards from PDF, text, and Markdown files using Claude AI.

## Features

- **Quality Assurance Pipeline**: 6-dimension confidence scoring + LLM critique for automatic improvement
- **Wozniak's 20 Rules of Knowledge Formulation**: Card generation based on cognitive science
- **8 Card Types**: QA, term definition, summary, cloze, reversible, sequence, compare & contrast, image occlusion
- **Bloom's Taxonomy**: Every card is tagged with a cognitive level (remember through create)
- **Cost Tracking**: Per-session API cost monitoring with configurable budget limits
- **Batch Processing**: Process entire directories of files in a single command
- **OCR Support**: Optional OCR fallback for image-heavy PDFs (via ocrmypdf)

## Installation

```bash
git clone https://github.com/shimo4228/Anki-QA.git
cd Anki-QA
uv sync --all-extras
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Claude API key |
| `PDF2ANKI_MODEL` | No | `claude-sonnet-4-5-20250929` | Override Claude model |
| `PDF2ANKI_BUDGET_LIMIT` | No | `1.00` | API cost budget limit (USD) |

## Usage

### Convert

```bash
# Basic conversion (PDF → TSV)
pdf2anki convert input.pdf

# Specify output file
pdf2anki convert input.pdf -o output.tsv

# JSON output
pdf2anki convert input.pdf --format json

# Both TSV and JSON
pdf2anki convert input.pdf --format both

# Full quality assurance pipeline
pdf2anki convert input.pdf --quality full

# Skip quality assurance
pdf2anki convert input.pdf --quality off

# Process entire directory
pdf2anki convert ./docs/

# Add custom tags and focus topics
pdf2anki convert input.pdf --tags "chapter1,important" --focus "machine learning"

# Limit card count and budget
pdf2anki convert input.pdf --max-cards 20 --budget-limit 0.50

# Enable OCR for image-heavy PDFs
pdf2anki convert input.pdf --ocr --lang jpn+eng

# Use custom config file
pdf2anki convert input.pdf --config my-config.yaml

# Debug logging
pdf2anki convert input.pdf --verbose
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | auto | Output file or directory |
| `--format` | `tsv` | Output format: `tsv`, `json`, `both` |
| `--quality` | `basic` | QA level: `off`, `basic`, `full` |
| `--model` | from config | Claude model name |
| `--max-cards` | `50` | Maximum cards to generate |
| `--tags` | - | Additional tags (comma-separated) |
| `--focus` | - | Focus topics (comma-separated) |
| `--card-types` | all 7 | Card types to generate (comma-separated) |
| `--bloom-filter` | all | Bloom levels to include (comma-separated) |
| `--budget-limit` | `1.00` | Budget limit in USD |
| `--ocr` | off | Enable OCR |
| `--lang` | `jpn+eng` | OCR language |
| `--config` | `config.yaml` | Path to config YAML |
| `--verbose` | off | Debug logging |

### Preview

Dry-run text extraction without API calls.

```bash
pdf2anki preview input.pdf
pdf2anki preview input.pdf --ocr
```

## Configuration

Settings are loaded with priority: **env vars > config.yaml > defaults**.

See [`config.yaml`](config.yaml) for all options including model, quality thresholds, card types, cost limits, and OCR settings.

## Architecture

```
[Input] PDF / TXT / MD
  ↓
[Step 1] Text Extraction (pymupdf4llm + OCR fallback)
  ↓
[Step 2] LLM Structured Extraction (Claude API Structured Outputs)
  ↓
[Step 3] Quality Assurance (6-dim Confidence Score → LLM Critique)
  ↓
[Step 4] Output (TSV / JSON)
```

### Quality Pipeline

Cards are scored across 6 dimensions (front quality, back quality, card type fit, bloom level fit, tags quality, atomicity). Cards below the confidence threshold are sent through LLM critique for improvement, splitting, or removal.

## Project Structure

```
src/pdf2anki/
  main.py        # CLI (typer): convert, preview commands
  config.py      # YAML + env var config loader
  schemas.py     # Pydantic models (AnkiCard, ExtractionResult, etc.)
  extract.py     # Text extraction (pymupdf4llm + OCR)
  structure.py   # LLM structured card extraction
  prompts.py     # Wozniak-based prompt templates
  quality.py     # Quality assurance pipeline
  convert.py     # TSV/JSON output conversion
  cost.py        # API cost tracking
tests/           # 266 tests (80%+ coverage required)
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Anthropic API key

## Docs

- [Contributing Guide](docs/CONTRIB.md) - Development workflow, testing, CLI reference
- [Runbook](docs/RUNBOOK.md) - Deployment, troubleshooting, common issues

## License

AGPL-3.0 License - See [LICENSE](LICENSE) for details.
