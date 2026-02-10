# Runbook - pdf2anki

## Deployment

### Local Installation

```bash
# Install from source
git clone https://github.com/shimo4228/pdf2anki.git
cd pdf2anki
uv sync --all-extras

# Verify installation
pdf2anki --help
```

### Environment Configuration

1. Copy `.env.example` to `.env`
2. Set `ANTHROPIC_API_KEY` (required)
3. Optionally override model via `PDF2ANKI_MODEL`
4. Optionally set budget via `PDF2ANKI_BUDGET_LIMIT`

### Config File

Default config: `config.yaml` in project root. Override with `--config` flag.

Key settings to verify before production use:
- `model`: Claude model to use
- `quality.confidence_threshold`: 0.90 (default)
- `cost.budget_limit`: 1.00 USD (default)

## Monitoring

### Cost Tracking

pdf2anki tracks API costs per session. After each run, the summary table shows:
- Total API calls made
- Total cost in USD
- Cards generated / passed / critiqued / removed

### Budget Alerts

- **Warning**: Triggered at 80% of budget (configurable via `cost.warn_at`)
- **Hard limit**: Processing stops when budget exceeded (configurable via `cost.budget_limit` or `--budget-limit`)

### Verbose Logging

Enable debug output for troubleshooting:

```bash
pdf2anki convert input.pdf --verbose
```

## Common Issues and Fixes

### 1. `ANTHROPIC_API_KEY not configured`

**Cause**: Missing API key.

**Fix**:
```bash
# Set in .env file
echo "ANTHROPIC_API_KEY=sk-ant-xxxxx" > .env

# Or export directly
export ANTHROPIC_API_KEY=sk-ant-xxxxx
```

### 2. `Unsupported file type`

**Cause**: Input file is not PDF, TXT, or MD.

**Fix**: Convert to a supported format, or extract text manually and save as `.txt`.

### 3. `No supported files found in <directory>`

**Cause**: Directory contains no PDF/TXT/MD files.

**Fix**: Verify the directory path and that it contains supported files.

### 4. `Config file not found`

**Cause**: `--config` points to a non-existent file.

**Fix**: Verify the path or omit `--config` to use defaults.

### 5. OCR Failures

**Cause**: `ocrmypdf` not installed or language pack missing.

**Fix**:
```bash
# Install OCR extra
uv sync --extra ocr

# Install Tesseract (macOS)
brew install tesseract tesseract-lang

# Verify OCR works
pdf2anki preview input.pdf --ocr
```

### 6. Budget Exceeded

**Cause**: API costs hit the budget limit mid-processing.

**Fix**:
```bash
# Increase budget
pdf2anki convert input.pdf --budget-limit 5.00

# Or reduce scope
pdf2anki convert input.pdf --max-cards 10 --quality off
```

### 7. Low Quality Cards

**Cause**: Quality pipeline is off or threshold is too low.

**Fix**:
```bash
# Enable full quality pipeline
pdf2anki convert input.pdf --quality full

# Or adjust threshold in config.yaml
# quality:
#   confidence_threshold: 0.95
#   max_critique_rounds: 3
```

### 8. Test Failures

```bash
# Run full test suite with verbose output
uv run pytest -v

# Run with coverage to find untested code
uv run pytest --cov=pdf2anki --cov-report=term-missing

# Run single failing test
uv run pytest tests/test_main.py::test_name -v
```

## Rollback Procedures

### Revert to Previous Version

```bash
# Check available versions
git log --oneline

# Revert to specific commit
git checkout <commit-hash>

# Reinstall dependencies
uv sync --all-extras
```

### Revert Configuration

```bash
# Restore default config
git checkout config.yaml

# Reset environment
cp .env.example .env
```

## Architecture Overview

```
Input (PDF/TXT/MD)
  |
  v
[extract.py] Text Extraction
  |  pymupdf4llm for PDF, plain read for TXT/MD
  |  Optional OCR fallback (ocrmypdf)
  v
[structure.py] LLM Structured Extraction
  |  Claude API with structured outputs
  |  Wozniak 20 rules prompts (prompts.py)
  |  Cost tracked (cost.py)
  v
[quality.py] Quality Assurance Pipeline
  |  Step 1: Confidence scoring (6 dimensions)
  |  Step 2: LLM critique for low-confidence cards
  |  Step 3: Apply improvements/splits/removals
  v
[convert.py] Output
  |  TSV: Anki-importable format with headers
  |  JSON: Full metadata with schema version
  v
Summary (rich table)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| anthropic | Claude API client |
| pymupdf / pymupdf4llm | PDF text extraction |
| pydantic | Data validation and schemas |
| pyyaml | YAML config parsing |
| typer | CLI framework |
| rich | Terminal output formatting |
| ocrmypdf (optional) | OCR for image-heavy PDFs |
