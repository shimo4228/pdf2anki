Language: English | [日本語](README.ja.md)

# pdf2anki

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-green)](LICENSE)

pdf2anki is a command-line tool that turns PDF, text, and Markdown files into Anki cards with Anthropic's Claude, for people who study from documents with Anki. By default it scores every card before export, and with `--quality full` it sends low-scoring cards through an LLM critique that improves, splits, or removes them.

Built as a hands-on project to test defenses against what LLM card generation produces (broken JSON, uneven card quality, and invisible cost) by combining LLM structured extraction, cognitive science (Wozniak's 20 Rules, the card-writing rules from the creator of SuperMemo), and a keyword-matching evaluation of prompt changes.

> [!NOTE]
> `convert` and the web UI send your document text (and, with `--vision`, the images embedded in image-heavy pages) to the Anthropic API, `eval` sends the texts of its labeled dataset, and all three need your own paid `ANTHROPIC_API_KEY`. Each run stops at a budget limit, 1.00 USD by default. `preview` makes no API calls.

The story behind the tool and a later audit of its code are in [More from the author](#more-from-the-author).

## Installation

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), and an Anthropic API key (paid). OCR (`--ocr`) also needs Tesseract and its language packs on the system ([Runbook](docs/RUNBOOK.md)).

```bash
git clone https://github.com/shimo4228/pdf2anki.git
cd pdf2anki
uv sync --all-extras
```

`uv sync` installs the `pdf2anki` command into `.venv`. Run it as `uv run pdf2anki ...`, or activate the environment once with `source .venv/bin/activate`; the examples below assume it is active.

### Environment Setup

pdf2anki reads the API key and the two `PDF2ANKI_` variables below from the environment; it does not load a `.env` file. Export the key in the shell you run it from:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

If you keep the variables in a `.env` file (copied from `.env.example`), load it into the shell first with `set -a; source .env; set +a`, or run through `uv run --env-file .env pdf2anki ...`.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Claude API key |
| `PDF2ANKI_MODEL` | No | `claude-sonnet-4-5-20250929` | Override Claude model |
| `PDF2ANKI_BUDGET_LIMIT` | No | `1.00` | API cost budget limit (USD) |

### First run

```bash
pdf2anki preview input.pdf     # text extraction only, no API call
pdf2anki convert input.pdf     # writes input.tsv next to the PDF
```

`convert` prints a summary table (cards generated, API calls, total cost, output path). The TSV has one card per row (front, back, tags, note type; a reversible card takes two rows, forward and reverse) with Anki import headers (`#separator:tab`, `#html:true`, and which columns hold tags and note type), so you can import it with File > Import in Anki, or send the cards straight to a running Anki with `--push` (needs the AnkiConnect add-on, which lets other programs add cards to a running Anki).

## Features

- **Quality Assurance Pipeline**: at the default level (`--quality basic`), each card is scored on 6 dimensions (front quality, back quality, card type fit, Bloom level fit, tags quality, atomicity, meaning one fact per card) and likely duplicates are flagged, with no extra API call; `--quality full` also sends cards whose weighted score falls below the confidence threshold (0.90 on a 0 to 1 scale by default) through an LLM critique, a second Claude call that improves, splits, or removes them, and `--quality off` skips both
- **Wozniak's 20 Rules of Knowledge Formulation**: the prompts apply the minimum information principle and ask for one concept per card
- **8 Card Types**: QA, term definition, summary, cloze, reversible, sequence, compare & contrast, image occlusion (a text description of a diagram with one region hidden, asking what belongs there; the image itself is not embedded in the card, and the `--vision` prompt asks for these cards)
- **Bloom's Taxonomy**: Every card is tagged with a cognitive level (remember through create), and `--bloom-filter` keeps only the levels you choose
- **Section-Aware Processing**: documents are split at headings, and each section goes to Claude with its heading path (for example `Book > Part > Chapter`) as context, which also becomes the cards' hierarchical tags

## Usage

### Convert

Turn PDF, text, and Markdown files into cards.

```bash
# Basic conversion (PDF → TSV)
pdf2anki convert input.pdf

# Specify output file
pdf2anki convert input.pdf -o output.tsv

# Both TSV and JSON
pdf2anki convert input.pdf --format both

# Process entire directory
pdf2anki convert ./docs/

# Add custom tags and focus topics
pdf2anki convert input.pdf --tags "chapter1,important" --focus "machine learning"

# Limit card count and budget
pdf2anki convert input.pdf --max-cards 20 --budget-limit 0.50

# Combine options
pdf2anki convert input.pdf --cache --vision --review --quality full --format both
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | auto | Output file or directory |
| `--format` | `tsv` | Output format: `tsv`, `json`, `both` |
| `--quality` | `basic` | QA level: `off`, `basic`, `full` |
| `--model` | auto (Sonnet, or Haiku for short sections) | Claude model name |
| `--max-cards` | `50` | Maximum cards to generate |
| `--tags` | - | Additional tags (comma-separated) |
| `--focus` | - | Focus topics (comma-separated) |
| `--card-types` | `qa,term_definition,summary_point,reversible,sequence,compare_contrast` | Card types to generate (comma-separated); add `image_occlusion` to ask for image occlusion cards, which `--vision` also asks for |
| `--bloom-filter` | all | Bloom levels to include (comma-separated) |
| `--budget-limit` | `1.00` | Budget limit in USD |
| `--review` | off | Review, accept, reject, and edit cards in a terminal UI before export |
| `--vision` / `--no-vision` | off | Generate text cards about the images embedded in image-heavy pages, through the Claude Vision API (sends those images to the API; also writes the extracted images to a `_media/` folder beside the output, which the TSV does not reference) |
| `--cache` / `--no-cache` | off | Skip re-extracting files whose content is unchanged (SHA-256 hash) |
| `--cache-clear` | off | Clear the extraction cache before processing |
| `--batch` | off | Use Batch API (50% discount, async) |
| `--ocr` | off | OCR fallback for image-heavy PDFs (ocrmypdf; needs Tesseract) |
| `--lang` | `jpn+eng` | OCR language |
| `--push` | off | Push cards to a running Anki desktop through the AnkiConnect add-on (local, `127.0.0.1:8765`) |
| `--deck` | `pdf2anki` | Anki deck name for `--push` |
| `--config` | none (built-in settings) | Path to a config YAML, such as the bundled `config.yaml` |
| `--verbose` | off | Debug logging |

### Preview

Dry-run text extraction without API calls.

```bash
pdf2anki preview input.pdf
pdf2anki preview input.pdf --ocr
```

### Eval

Measure prompt quality against a labeled dataset: each expected card lists keywords, a generated card matches it when enough of them appear, and the matches are scored as Recall, Precision, and F1. This measures keyword coverage, not how well a card teaches.

```bash
# Run evaluation
pdf2anki eval --dataset evals/dataset.yaml

# Output JSON report
pdf2anki eval --dataset evals/dataset.yaml --output eval-report.json
```

### Web UI

A Gradio interface for the same conversion, installed by `uv sync --all-extras` (the `web` extra).

```bash
pdf2anki web                   # http://127.0.0.1:7860
```

The web UI has no authentication. Binding it to another host with `--host` lets anyone on that network spend your API budget.

## Configuration

Settings are loaded with priority: **CLI flags > env vars > config YAML > defaults**. A config YAML is read only when you pass it with `--config`; without it, the built-in defaults apply.

The bundled [`config.yaml`](config.yaml) covers model, quality thresholds, output, card types (it adds cloze to the built-in default), cost limits, OCR, and cache settings. Start from it with `pdf2anki convert input.pdf --config config.yaml`.

## Docs

- [Contributing Guide](docs/CONTRIB.md) - Development workflow and testing
- [Runbook](docs/RUNBOOK.md) - Deployment, troubleshooting, common issues

## More from the author

- **[Never Trust LLM Output — 6 Defenses from Building a PDF-to-Anki CLI](https://dev.to/shimo4228/never-trust-llm-output-6-defenses-from-building-a-pdf-to-anki-cli-43mo)** ([日本語](https://zenn.dev/shimo4228/articles/ecc-journey-part2)): why pdf2anki validates each card separately, scores cards before the LLM critique, caps cost, and splits by section, told through what broke while building it.
- **[Can Six-Month-Old AI Code Survive Today's Review? A 25-Bug Triage](https://dev.to/shimo4228/can-six-month-old-ai-code-survive-todays-review-a-25-bug-triage-4o6d)** ([日本語](https://zenn.dev/shimo4228/articles/ai-code-half-year-audit)): in August 2026, Claude Fable 5 and GPT-5.6-Sol (through Codex CLI) audited this codebase, written about six months earlier; the design held, while pushing reversible cards to Anki and extracting large images had never worked before the fixes that followed.
- **[code-and-llm-collaboration](https://github.com/shimo4228/code-and-llm-collaboration)**: four patterns for layering deterministic code and LLM calls in one pipeline; pdf2anki's heuristic scoring followed by LLM critique is one of them.
- **[llm-as-judge](https://github.com/shimo4228/llm-as-judge)**: how to design an LLM judge with binary checks as evidence and one named verdict, if you want to take the critique step further.
- **[shimo4228](https://github.com/shimo4228/shimo4228)**: the author's hub, with five long-running practice lines, their DOIs, and the tools for Claude Code.

## License

AGPL-3.0 License - See [LICENSE](LICENSE) for details.

<details>
<summary>For tools and AI assistants</summary>

pdf2anki is a command-line tool that turns PDF, plain-text, and Markdown files into Anki cards with Anthropic's Claude API, for learners who study from documents with Anki. It exists because LLM card generation produces broken JSON, uneven card quality, and invisible cost, and the author built it to test defenses against each: per-card schema validation that accepts partial success, card rules based on Wozniak's 20 Rules of Knowledge Formulation, a quality pipeline, a hard budget limit, and a keyword-matching evaluation of prompt changes.

Canonical facts: license AGPL-3.0 (`AGPL-3.0-or-later` in `pyproject.toml`); Python 3.12+ managed with uv; CLI built on Typer and Rich, review TUI on Textual, optional web UI on Gradio, PDF extraction with PyMuPDF and pymupdf4llm, optional OCR with ocrmypdf. Status: a hands-on project, curated by hand by the author in this repository; its latest changes are the fixes that followed the August 2026 code audit. It is a standalone tool, not part of one of the author's practice lines. It requires a paid `ANTHROPIC_API_KEY`; `convert` and the web UI send document text, and with `--vision` the images embedded in image-heavy pages, to the Anthropic API, `eval` sends the texts of its labeled dataset, and `preview` runs locally. The default model is `claude-sonnet-4-5-20250929`, sections under 10,000 characters that ask for fewer than 30 cards are routed to Claude Haiku automatically unless a model is set explicitly, and each run stops at a budget limit of 1.00 USD unless configured otherwise. `--push` talks only to a local Anki through AnkiConnect at `127.0.0.1:8765`.

Each card has a front, a back, one of 8 card types (qa, term_definition, summary_point, cloze, reversible, sequence, compare_contrast, image_occlusion), a Bloom's taxonomy level, and tags. `convert` runs text extraction (pymupdf4llm, with OCR fallback and an optional cache), heading-based section splitting, structured card extraction with Claude (with `--vision` on a PDF, one Claude Vision request with the whole text and images replaces the section-split extraction and skips the Haiku routing), the quality pipeline, optional review in the TUI, and TSV or JSON output. At the default level (`--quality basic`) the quality pipeline scores cards on 6 dimensions and flags likely duplicates; `--quality full` also sends cards whose weighted 6-dimension score falls below the confidence threshold (0.90 by default) to an LLM critique that improves, splits, or removes them; `--quality off` skips it.

Example: `pdf2anki convert input.pdf --quality full` writes `input.tsv` next to the PDF, one card per row (two for a reversible card, forward and reverse) as front, back, tags, and note type, with Anki import headers and a `bloom::<level>` tag on each card, and prints a summary of cards generated, API calls, total cost, cards passed, critiqued, and removed, and the output path.

Pointers: [`config.yaml`](config.yaml) is a sample settings file, read only when passed with `--config`, and has no vision or Batch API section; [docs/CONTRIB.md](docs/CONTRIB.md) covers development and testing; [docs/RUNBOOK.md](docs/RUNBOOK.md) covers troubleshooting (including OCR); [evals/dataset.yaml](evals/dataset.yaml) is the labeled set used by `pdf2anki eval`. The author's other work is indexed at https://github.com/shimo4228/shimo4228.

</details>
