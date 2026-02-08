# pdf2anki

[日本語版はこちら / Japanese](README.ja.md)

A CLI tool that automatically generates high-quality Anki flashcards from PDF, text, and Markdown files using Claude AI.

## Features

- **Quality Assurance Pipeline**: Confidence scoring + LLM critique for automatic improvement
- **Wozniak's 20 Rules of Knowledge Formulation**: Card generation based on cognitive science
- **8 Card Types**: QA, term definition, summary, cloze, reversible, sequence, compare & contrast, image occlusion
- **Bloom's Taxonomy**: Every card is tagged with a cognitive level (remember through create)

## Installation

```bash
git clone https://github.com/shimo4228/Anki-QA.git
cd Anki-QA
uv sync --all-extras
```

## Usage

```bash
# Basic conversion (PDF → TSV)
pdf2anki convert input.pdf

# Specify output file
pdf2anki convert input.pdf -o output.tsv

# Enable full quality assurance pipeline
pdf2anki convert input.pdf --quality full

# Preview (text extraction only)
pdf2anki preview input.pdf
```

## Architecture

```
[Input] PDF / TXT / MD
  ↓
[Step 1] Text Extraction (pymupdf4llm + OCR fallback)
  ↓
[Step 2] LLM Structured Extraction (Claude API Structured Outputs)
  ↓
[Step 3] Quality Assurance (Confidence Score → LLM Critique)
  ↓
[Step 4] Output (TSV / JSON)
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Anthropic API key

## License

MIT License
