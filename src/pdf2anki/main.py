"""CLI entry point for pdf2anki.

Provides three subcommands:
  - convert: Extract text -> generate cards -> quality check -> output TSV/JSON
  - preview: Dry-run text extraction (no API calls)
  - eval: Evaluate prompt quality against a labeled dataset
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from pdf2anki.config import AppConfig, load_config
from pdf2anki.cost import CostTracker, estimate_cost
from pdf2anki.extract import estimate_tokens
from pdf2anki.quality import QualityReport
from pdf2anki.service import (
    collect_files,
    extract_with_cache,
    merge_quality_reports,
    process_file,
    resolve_output_path,
    write_output,
)

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
    cache: bool | None,
    vision: bool | None = None,
) -> AppConfig:
    """Build AppConfig from base config + CLI overrides."""
    base = load_config(config_path)

    overrides: dict[str, Any] = {}

    if model is not None:
        overrides["model"] = model
        overrides["model_overridden"] = True
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
    if cache is not None:
        overrides["cache_enabled"] = cache
    if vision is not None:
        overrides["vision_enabled"] = vision

    if quality == QualityLevel.OFF:
        overrides["quality_enable_critique"] = False
        overrides["quality_confidence_threshold"] = 0.0
    elif quality == QualityLevel.FULL:
        overrides["quality_enable_critique"] = True

    if not overrides:
        return base

    return base.model_copy(update=overrides)


def _print_summary(
    *,
    card_count: int,
    cost_tracker: CostTracker,
    quality_report: QualityReport | None,
    written_files: list[Path],
    pushed_count: int = 0,
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

    if pushed_count > 0:
        table.add_row("Pushed to Anki", str(pushed_count))

    console.print(table)


def _parse_csv_option(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI option into a list, or None."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",")]


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
    batch: bool = typer.Option(
        False, "--batch", help="Use Batch API for 50% cost savings"
    ),
    cache: bool | None = typer.Option(
        None, "--cache/--no-cache", help="Enable/disable extraction cache"
    ),
    cache_clear: bool = typer.Option(
        False, "--cache-clear", help="Clear extraction cache before processing"
    ),
    vision: bool | None = typer.Option(
        None,
        "--vision/--no-vision",
        help="Enable/disable Vision API for image-heavy PDFs",
    ),
    review: bool = typer.Option(
        False, "--review", help="Launch interactive review TUI before saving"
    ),
    push: bool = typer.Option(
        False, "--push", help="Push cards to Anki via AnkiConnect"
    ),
    deck: str = typer.Option(
        "pdf2anki", "--deck", help="Anki deck name for --push"
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
            cache=cache,
            vision=vision,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e

    if cache_clear:
        from pdf2anki.cache import invalidate_cache as _invalidate

        count = _invalidate(Path(config.cache_dir))
        console.print(f"[yellow]Cache cleared:[/yellow] {count} entries removed")

    path = Path(input_path)
    try:
        files = collect_files(path)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e

    is_dir_input = path.is_dir()
    output_path = resolve_output_path(path, output, fmt.value, is_dir_input)
    additional_tags = _parse_csv_option(tags)
    focus_topics = _parse_csv_option(focus)

    cost_tracker = CostTracker(budget_limit=config.cost_budget_limit)
    all_written: list[Path] = []
    all_reports: list[QualityReport] = []
    total_cards = 0
    total_pushed = 0

    for file_path in files:
        console.print(f"Processing: [cyan]{file_path.name}[/cyan]")
        try:
            result, report, cost_tracker = process_file(
                file_path=file_path,
                config=config,
                cost_tracker=cost_tracker,
                quality=quality.value,
                focus_topics=focus_topics,
                additional_tags=additional_tags,
                batch=batch,
                output_dir=output_path if output_path.is_dir() else output_path.parent,
            )
        except (RuntimeError, ValueError) as e:
            console.print(f"[red]Error processing {file_path.name}:[/red] {e}")
            continue

        if report is not None:
            all_reports.append(report)

        if review and result.cards:
            from pdf2anki.quality.heuristic import score_cards
            from pdf2anki.tui import launch_review

            scores = score_cards(list(result.cards))
            result = launch_review(result, scores)

        written = write_output(
            result=result, output_path=output_path, fmt=fmt.value,
            source_stem=file_path.stem, additional_tags=additional_tags,
        )
        all_written.extend(written)
        total_cards += result.card_count

        if push and result.cards:
            from pdf2anki.anki_connect import push_cards

            push_result = push_cards(list(result.cards), deck_name=deck)
            total_pushed += push_result.added
            if push_result.failed > 0:
                console.print(
                    f"[yellow]Warning:[/yellow] {push_result.failed} card(s) "
                    f"failed to push"
                )

    quality_report = merge_quality_reports(all_reports) if all_reports else None

    _print_summary(
        card_count=total_cards, cost_tracker=cost_tracker,
        quality_report=quality_report, written_files=all_written,
        pushed_count=total_pushed,
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
    cache: bool | None = typer.Option(
        None, "--cache/--no-cache", help="Enable/disable extraction cache"
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

    config = AppConfig(
        ocr_enabled=ocr,
        ocr_lang=ocr_lang,
        cache_enabled=cache if cache is not None else False,
    )
    doc = extract_with_cache(path, config=config)

    table = Table(title="Preview: Text Extraction", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    tokens = estimate_tokens(doc.text)

    table.add_row("Source", doc.source_path)
    table.add_row("File type", doc.file_type)
    table.add_row("OCR used", str(doc.used_ocr))
    table.add_row("Text length", f"{len(doc.text)} chars")
    table.add_row("Est. tokens", f"{tokens:,}")
    cost = estimate_cost("claude-sonnet-4-5-20250929", tokens, tokens // 4)
    table.add_row("Est. cost", f"${cost:.4f}")
    table.add_row("Chunks", str(len(doc.chunks)))

    console.print(table)
    console.print("\n[bold]Extracted text (first 500 chars):[/bold]")
    console.print(doc.text[:500])

    if len(doc.chunks) > 1:
        console.print(f"\n[dim]({len(doc.chunks)} chunks total)[/dim]")


@app.command(name="eval")
def eval_cmd(
    dataset: str = typer.Option(
        ..., "--dataset", help="Path to evaluation dataset YAML"
    ),
    output: str | None = typer.Option(
        None, "-o", "--output", help="JSON report output path"
    ),
    model: str | None = typer.Option(
        None, "--model", help="Claude model name"
    ),
    config_path: str | None = typer.Option(
        None, "--config", help="Path to config YAML file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Enable debug logging"
    ),
) -> None:
    """Evaluate prompt quality against a labeled dataset."""
    _log_fmt = "%(name)s %(levelname)s: %(message)s"
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=_log_fmt)

    from pdf2anki.eval.dataset import load_dataset
    from pdf2anki.eval.matcher import match_cards
    from pdf2anki.eval.metrics import calculate_metrics
    from pdf2anki.eval.report import (
        print_eval_report,
        write_eval_json,
    )
    from pdf2anki.structure import extract_cards

    ds_path = Path(dataset)
    try:
        ds = load_dataset(ds_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e

    try:
        config = _build_config(
            config_path=config_path,
            model=model,
            max_cards=None,
            card_types=None,
            bloom_filter=None,
            budget_limit=None,
            ocr=False,
            lang=None,
            quality=QualityLevel.OFF,
            cache=False,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e

    cost_tracker = CostTracker(budget_limit=config.cost_budget_limit)
    case_results = []

    console.print(
        f"[bold]Evaluating:[/bold] {ds.name} v{ds.version} "
        f"({len(ds.cases)} cases)"
    )

    for case in ds.cases:
        console.print(f"  Case: [cyan]{case.id}[/cyan] ", end="")
        try:
            result, cost_tracker = extract_cards(
                text=case.text,
                source_file=f"eval:{case.id}",
                config=config,
                cost_tracker=cost_tracker,
            )
            cr = match_cards(
                list(case.expected_cards),
                list(result.cards),
                case_id=case.id,
            )
            case_results.append(cr)
            matched = sum(
                1 for m in cr.matches if m.matched_card is not None
            )
            console.print(
                f"[green]{matched}/{len(case.expected_cards)} matched[/green]"
            )
        except (RuntimeError, ValueError) as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    metrics = calculate_metrics(
        case_results, cost_usd=cost_tracker.total_cost
    )
    print_eval_report(metrics, case_results)

    if output is not None:
        out_path = Path(output)
        write_eval_json(metrics, case_results, out_path)
        console.print(f"\n[green]Report saved:[/green] {out_path}")
