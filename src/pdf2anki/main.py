"""CLI entry point for pdf2anki.

Provides two subcommands:
  - convert: Extract text -> generate cards -> quality check -> output TSV/JSON
  - preview: Dry-run text extraction (no API calls)
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pdf2anki.config import AppConfig, load_config
from pdf2anki.convert import write_json, write_tsv
from pdf2anki.cost import CostTracker
from pdf2anki.extract import extract_text
from pdf2anki.quality import QualityReport, run_quality_pipeline
from pdf2anki.schemas import ExtractionResult
from pdf2anki.structure import extract_cards

app = typer.Typer(
    name="pdf2anki",
    help="Generate high-quality Anki flashcards from PDF/TXT/MD using Claude AI.",
    no_args_is_help=True,
)

console = Console()

_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
_DEFAULT_OCR_LANG = "jpn+eng"


class OutputFormat(StrEnum):
    TSV = "tsv"
    JSON = "json"
    BOTH = "both"


class QualityLevel(StrEnum):
    OFF = "off"
    BASIC = "basic"
    FULL = "full"


def _collect_files(input_path: Path) -> list[Path]:
    """Collect supported files from a path (file or directory)."""
    if input_path.is_file():
        if input_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise typer.BadParameter(
                f"Unsupported file type: {input_path.suffix}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )
        return [input_path]

    if input_path.is_dir():
        files = [
            f
            for f in sorted(input_path.iterdir())
            if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
        ]
        if not files:
            raise typer.BadParameter(f"No supported files found in {input_path}")
        return files

    raise typer.BadParameter(f"Path not found: {input_path}")


def _build_config(
    *,
    config_path: str | None,
    model: str | None,
    max_cards: int | None,
    card_types: str | None,
    bloom_filter: str | None,
    budget_limit: float | None,
    ocr: bool,
    lang: str | None,
    quality: QualityLevel,
) -> AppConfig:
    """Build AppConfig from base config + CLI overrides."""
    base = load_config(config_path)

    overrides: dict = {}

    if model is not None:
        overrides["model"] = model
    if max_cards is not None:
        overrides["cards_max_cards"] = max_cards
    if card_types is not None:
        overrides["cards_card_types"] = [t.strip() for t in card_types.split(",")]
    if bloom_filter is not None:
        overrides["cards_bloom_filter"] = [b.strip() for b in bloom_filter.split(",")]
    if budget_limit is not None:
        overrides["cost_budget_limit"] = budget_limit
    if ocr:
        overrides["ocr_enabled"] = True
    if lang is not None:
        overrides["ocr_lang"] = lang

    if quality == QualityLevel.OFF:
        overrides["quality_enable_critique"] = False
        overrides["quality_confidence_threshold"] = 0.0
    elif quality == QualityLevel.FULL:
        overrides["quality_enable_critique"] = True

    if not overrides:
        return base

    return base.model_copy(update=overrides)


def _resolve_output_path(
    input_path: Path,
    output: str | None,
    fmt: OutputFormat,
    is_directory_input: bool,
) -> Path:
    """Determine the output path."""
    if output is not None:
        return Path(output)

    if is_directory_input:
        return input_path.parent / "output"

    if fmt == OutputFormat.BOTH:
        return input_path.parent

    return input_path.with_suffix(f".{fmt.value}")


def _write_output(
    *,
    result: ExtractionResult,
    output_path: Path,
    fmt: OutputFormat,
    source_stem: str,
    additional_tags: list[str] | None,
) -> list[Path]:
    """Write cards to the requested format(s). Returns list of written files."""
    written: list[Path] = []

    if fmt in (OutputFormat.TSV, OutputFormat.BOTH):
        if output_path.suffix == ".tsv":
            tsv_path = output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            tsv_path = output_path / f"{source_stem}.tsv"
        write_tsv(list(result.cards), tsv_path, additional_tags)
        written.append(tsv_path)

    if fmt in (OutputFormat.JSON, OutputFormat.BOTH):
        if output_path.suffix == ".json":
            json_path = output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            json_path = output_path / f"{source_stem}.json"
        write_json(result, json_path)
        written.append(json_path)

    return written


def _merge_quality_reports(reports: list[QualityReport]) -> QualityReport:
    """Merge multiple QualityReports into a single aggregate report."""
    return QualityReport(
        total_cards=sum(r.total_cards for r in reports),
        passed_cards=sum(r.passed_cards for r in reports),
        critiqued_cards=sum(r.critiqued_cards for r in reports),
        removed_cards=sum(r.removed_cards for r in reports),
        improved_cards=sum(r.improved_cards for r in reports),
        split_cards=sum(r.split_cards for r in reports),
        final_card_count=sum(r.final_card_count for r in reports),
    )


def _print_summary(
    *,
    card_count: int,
    cost_tracker: CostTracker,
    quality_report: QualityReport | None,
    written_files: list[Path],
) -> None:
    """Print a rich summary table to console."""
    table = Table(title="pdf2anki Summary", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Cards generated", str(card_count))
    table.add_row("API calls", str(cost_tracker.request_count))
    table.add_row("Total cost", f"${cost_tracker.total_cost:.4f}")

    if quality_report is not None:
        table.add_row("QA passed", str(quality_report.passed_cards))
        table.add_row("QA critiqued", str(quality_report.critiqued_cards))
        table.add_row("QA removed", str(quality_report.removed_cards))

    for f in written_files:
        table.add_row("Output", str(f))

    console.print(table)


def _parse_csv_option(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI option into a list, or None."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",")]


def _process_file(
    *,
    file_path: Path,
    config: AppConfig,
    cost_tracker: CostTracker,
    quality: QualityLevel,
    focus_topics: list[str] | None,
    additional_tags: list[str] | None,
) -> tuple[ExtractionResult, QualityReport | None, CostTracker]:
    """Process a single file: extract -> generate cards -> quality check."""
    doc = extract_text(
        file_path,
        ocr_enabled=config.ocr_enabled,
        ocr_lang=config.ocr_lang,
    )

    bloom_filter = config.cards_bloom_filter or None

    result, cost_tracker = extract_cards(
        doc.text,
        source_file=str(file_path.name),
        config=config,
        cost_tracker=cost_tracker,
        chunks=list(doc.chunks) if len(doc.chunks) > 1 else None,
        focus_topics=focus_topics,
        bloom_filter=bloom_filter,
        additional_tags=additional_tags,
    )

    quality_report: QualityReport | None = None

    if quality != QualityLevel.OFF:
        cards, quality_report, cost_tracker = run_quality_pipeline(
            cards=list(result.cards),
            source_text=doc.text,
            config=config,
            cost_tracker=cost_tracker,
        )
        result = ExtractionResult(
            source_file=result.source_file,
            cards=cards,
            model_used=result.model_used,
        )

    return result, quality_report, cost_tracker


@app.command()
def convert(
    input_path: str = typer.Argument(
        ..., help="Input file or directory (PDF/TXT/MD)"
    ),
    output: str | None = typer.Option(
        None, "-o", "--output", help="Output file or directory"
    ),
    fmt: OutputFormat = typer.Option(  # noqa: B008
        OutputFormat.TSV, "--format", help="Output format"
    ),
    quality: QualityLevel = typer.Option(  # noqa: B008
        QualityLevel.BASIC,
        "--quality",
        help="Quality pipeline level",
    ),
    model: str | None = typer.Option(
        None, "--model", help="Claude model name"
    ),
    max_cards: int | None = typer.Option(
        None, "--max-cards", help="Maximum cards to generate"
    ),
    tags: str | None = typer.Option(
        None, "--tags", help="Additional tags (comma-separated)"
    ),
    focus: str | None = typer.Option(
        None, "--focus", help="Focus topics (comma-separated)"
    ),
    card_types: str | None = typer.Option(
        None,
        "--card-types",
        help="Card types to generate (comma-separated)",
    ),
    bloom_filter: str | None = typer.Option(
        None,
        "--bloom-filter",
        help="Bloom levels to include (comma-separated)",
    ),
    budget_limit: float | None = typer.Option(
        None, "--budget-limit", help="Budget limit in USD"
    ),
    ocr: bool = typer.Option(
        False, "--ocr", help="Enable OCR for image-heavy PDFs"
    ),
    lang: str | None = typer.Option(
        None, "--lang", help="OCR language (default: jpn+eng)"
    ),
    config_path: str | None = typer.Option(
        None, "--config", help="Path to config YAML file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Enable debug logging"
    ),
) -> None:
    """Convert PDF/TXT/MD to Anki flashcards."""
    _log_fmt = "%(name)s %(levelname)s: %(message)s"
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=_log_fmt)

    try:
        config = _build_config(
            config_path=config_path,
            model=model,
            max_cards=max_cards,
            card_types=card_types,
            bloom_filter=bloom_filter,
            budget_limit=budget_limit,
            ocr=ocr,
            lang=lang,
            quality=quality,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e

    path = Path(input_path)
    try:
        files = _collect_files(path)
    except typer.BadParameter as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e

    is_dir_input = path.is_dir()
    output_path = _resolve_output_path(path, output, fmt, is_dir_input)
    additional_tags = _parse_csv_option(tags)
    focus_topics = _parse_csv_option(focus)

    cost_tracker = CostTracker(budget_limit=config.cost_budget_limit)
    all_written: list[Path] = []
    all_reports: list[QualityReport] = []
    total_cards = 0

    for file_path in files:
        console.print(f"Processing: [cyan]{file_path.name}[/cyan]")
        try:
            result, report, cost_tracker = _process_file(
                file_path=file_path,
                config=config,
                cost_tracker=cost_tracker,
                quality=quality,
                focus_topics=focus_topics,
                additional_tags=additional_tags,
            )
        except (RuntimeError, ValueError) as e:
            console.print(f"[red]Error processing {file_path.name}:[/red] {e}")
            continue

        if report is not None:
            all_reports.append(report)

        written = _write_output(
            result=result, output_path=output_path, fmt=fmt,
            source_stem=file_path.stem, additional_tags=additional_tags,
        )
        all_written.extend(written)
        total_cards += result.card_count

    quality_report = _merge_quality_reports(all_reports) if all_reports else None

    _print_summary(
        card_count=total_cards, cost_tracker=cost_tracker,
        quality_report=quality_report, written_files=all_written,
    )


@app.command()
def preview(
    input_path: str = typer.Argument(
        ..., help="Input file (PDF/TXT/MD)"
    ),
    ocr: bool = typer.Option(
        False, "--ocr", help="Enable OCR for image-heavy PDFs"
    ),
    lang: str | None = typer.Option(
        None, "--lang", help="OCR language (default: jpn+eng)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Enable debug logging"
    ),
) -> None:
    """Preview text extraction without generating cards (dry-run)."""
    _log_fmt = "%(name)s %(levelname)s: %(message)s"
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=_log_fmt)

    path = Path(input_path)
    if not path.is_file():
        console.print(f"[red]Error:[/red] File not found: {input_path}")
        raise typer.Exit(code=1)

    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        console.print(f"[red]Error:[/red] Unsupported file type: {path.suffix}")
        raise typer.Exit(code=1)

    ocr_lang = lang if lang else _DEFAULT_OCR_LANG

    doc = extract_text(path, ocr_enabled=ocr, ocr_lang=ocr_lang)

    table = Table(title="Preview: Text Extraction", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Source", doc.source_path)
    table.add_row("File type", doc.file_type)
    table.add_row("OCR used", str(doc.used_ocr))
    table.add_row("Text length", f"{len(doc.text)} chars")
    table.add_row("Chunks", str(len(doc.chunks)))

    console.print(table)
    console.print("\n[bold]Extracted text (first 500 chars):[/bold]")
    console.print(doc.text[:500])

    if len(doc.chunks) > 1:
        console.print(f"\n[dim]({len(doc.chunks)} chunks total)[/dim]")
