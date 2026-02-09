"""Service layer for pdf2anki business logic.

Separates orchestration logic from CLI concerns.
Functions here are independent of Typer/Rich and can be
called from any interface (CLI, future Web UI, tests).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pdf2anki.batch import (
    collect_batch_results,
    create_batch_requests,
    poll_batch,
    submit_batch,
)
from pdf2anki.cache import CacheEntry, compute_file_hash, read_cache, write_cache
from pdf2anki.config import AppConfig
from pdf2anki.convert import write_json, write_tsv
from pdf2anki.cost import CostRecord, CostTracker, estimate_cost
from pdf2anki.extract import ExtractedDocument, extract_text
from pdf2anki.quality import QualityReport, run_quality_pipeline
from pdf2anki.schemas import AnkiCard, ExtractionResult
from pdf2anki.section import Section
from pdf2anki.structure import extract_cards

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def collect_files(input_path: Path) -> list[Path]:
    """Collect supported files from a path (file or directory).

    Raises:
        ValueError: If path doesn't exist, file type is unsupported,
                    or directory contains no supported files.
    """
    if input_path.is_file():
        if input_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
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
            raise ValueError(f"No supported files found in {input_path}")
        return files

    raise ValueError(f"Path not found: {input_path}")


def resolve_output_path(
    input_path: Path,
    output: str | None,
    fmt: str,
    is_directory_input: bool,
) -> Path:
    """Determine the output path.

    Args:
        input_path: The original input file/directory path.
        output: Explicit output path from user, or None.
        fmt: Output format string ("tsv", "json", "both").
        is_directory_input: Whether the input was a directory.
    """
    if output is not None:
        return Path(output)

    if is_directory_input:
        return input_path.parent / "output"

    if fmt == "both":
        return input_path.parent

    return input_path.with_suffix(f".{fmt}")


def write_output(
    *,
    result: ExtractionResult,
    output_path: Path,
    fmt: str,
    source_stem: str,
    additional_tags: list[str] | None,
) -> list[Path]:
    """Write cards to the requested format(s). Returns list of written files.

    Args:
        fmt: Output format string ("tsv", "json", "both").
    """
    written: list[Path] = []

    if fmt in ("tsv", "both"):
        if output_path.suffix == ".tsv":
            tsv_path = output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            tsv_path = output_path / f"{source_stem}.tsv"
        write_tsv(list(result.cards), tsv_path, additional_tags)
        written.append(tsv_path)

    if fmt in ("json", "both"):
        if output_path.suffix == ".json":
            json_path = output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            json_path = output_path / f"{source_stem}.json"
        write_json(result, json_path)
        written.append(json_path)

    return written


def merge_quality_reports(reports: list[QualityReport]) -> QualityReport:
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


def extract_with_cache(
    file_path: Path,
    *,
    config: AppConfig,
) -> ExtractedDocument:
    """Extract text with optional caching.

    When cache is enabled, computes SHA-256 of the file and checks
    for a cached ExtractedDocument. On miss, runs extract_text() and
    stores the result. When disabled, delegates directly to extract_text().

    Args:
        file_path: Path to the input file.
        config: Application configuration (cache settings + OCR settings).

    Returns:
        ExtractedDocument from cache or fresh extraction.
    """
    if not config.cache_enabled:
        return extract_text(
            file_path,
            ocr_enabled=config.ocr_enabled,
            ocr_lang=config.ocr_lang,
        )

    cache_dir = Path(config.cache_dir)
    file_hash = compute_file_hash(file_path)

    cached = read_cache(cache_dir, file_hash)
    if cached is not None:
        logger.info("Cache hit: %s (hash=%s)", file_path.name, file_hash[:12])
        return cached.document

    logger.info("Cache miss: %s (hash=%s)", file_path.name, file_hash[:12])
    doc = extract_text(
        file_path,
        ocr_enabled=config.ocr_enabled,
        ocr_lang=config.ocr_lang,
    )

    entry = CacheEntry(
        file_hash=file_hash,
        source_path=str(file_path),
        document=doc,
    )
    write_cache(cache_dir, entry)

    return doc


def process_file(
    *,
    file_path: Path,
    config: AppConfig,
    cost_tracker: CostTracker,
    quality: str,
    focus_topics: list[str] | None,
    additional_tags: list[str] | None,
    batch: bool = False,
) -> tuple[ExtractionResult, QualityReport | None, CostTracker]:
    """Process a single file: extract -> generate cards -> quality check.

    Args:
        quality: Quality level string ("off", "basic", "full").
    """
    doc = extract_with_cache(file_path, config=config)

    bloom_filter = config.cards_bloom_filter or None
    sections_list = list(doc.sections) if doc.sections else None

    if batch and sections_list:
        result, cost_tracker = _process_file_batch(
            sections=sections_list,
            source_file=str(file_path.name),
            config=config,
            cost_tracker=cost_tracker,
            focus_topics=focus_topics,
            bloom_filter=bloom_filter,
            additional_tags=additional_tags,
        )
    else:
        result, cost_tracker = extract_cards(
            doc.text,
            source_file=str(file_path.name),
            config=config,
            cost_tracker=cost_tracker,
            chunks=list(doc.chunks) if len(doc.chunks) > 1 else None,
            sections=sections_list,
            focus_topics=focus_topics,
            bloom_filter=bloom_filter,
            additional_tags=additional_tags,
        )

    quality_report: QualityReport | None = None

    if quality != "off":
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


def _process_file_batch(
    *,
    sections: list[Section],
    source_file: str,
    config: AppConfig,
    cost_tracker: CostTracker,
    focus_topics: list[str] | None,
    bloom_filter: list[str] | None,
    additional_tags: list[str] | None,
) -> tuple[ExtractionResult, CostTracker]:
    """Process sections via Batch API (50% cost savings)."""
    requests = create_batch_requests(
        sections,
        document_title=source_file,
        config=config,
        focus_topics=focus_topics,
        bloom_filter=bloom_filter,
        additional_tags=additional_tags,
    )

    if not requests:
        return ExtractionResult(
            source_file=source_file,
            cards=[],
            model_used=config.model,
        ), cost_tracker

    batch_id = submit_batch(requests)
    logger.info("Batch submitted: %s", batch_id)

    poll_batch(
        batch_id,
        poll_interval=config.batch_poll_interval,
        timeout=config.batch_timeout,
    )

    batch_results = collect_batch_results(batch_id)

    all_cards: list[AnkiCard] = []
    model_used = ""
    for br in batch_results:
        all_cards.extend(br.cards)
        if not model_used:
            model_used = br.model

        cost = estimate_cost(
            model=br.model,
            input_tokens=br.input_tokens,
            output_tokens=br.output_tokens,
            batch=True,
        )
        record = CostRecord(
            model=br.model,
            input_tokens=br.input_tokens,
            output_tokens=br.output_tokens,
            cost_usd=cost,
        )
        cost_tracker = cost_tracker.add(record)

    result = ExtractionResult(
        source_file=source_file,
        cards=all_cards,
        model_used=model_used or config.model,
    )

    return result, cost_tracker
