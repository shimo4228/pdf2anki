"""Tests for pdf2anki service layer (service.py).

Covers:
- collect_files: file/directory collection with validation
- resolve_output_path: output path determination
- write_output: TSV/JSON/both output writing
- merge_quality_reports: report aggregation
- process_file: single file processing pipeline
- process_file_batch: batch API processing
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pdf2anki.cost import CostTracker
from pdf2anki.extract import ExtractedDocument
from pdf2anki.quality import QualityReport
from pdf2anki.schemas import AnkiCard, BloomLevel, CardType, ExtractionResult
from pdf2anki.section import Section
from pdf2anki.service import (
    collect_files,
    merge_quality_reports,
    process_file,
    resolve_output_path,
    write_output,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    p = tmp_path / "test.txt"
    p.write_text("テスト用テキスト。", encoding="utf-8")
    return p


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("文書A。", encoding="utf-8")
    (d / "b.md").write_text("# 文書B", encoding="utf-8")
    (d / "c.pdf").write_bytes(b"%PDF-1.4 fake")
    (d / "ignore.csv").write_text("col1,col2", encoding="utf-8")
    return d


@pytest.fixture
def mock_cards() -> list[AnkiCard]:
    return [
        AnkiCard(
            front="ReLUの数式は？",
            back="f(x) = max(0, x)",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI::活性化関数"],
        ),
    ]


@pytest.fixture
def mock_extraction_result(mock_cards: list[AnkiCard]) -> ExtractionResult:
    return ExtractionResult(
        source_file="test.txt",
        cards=mock_cards,
        model_used="claude-haiku-4-5-20251001",
    )


@pytest.fixture
def mock_extracted_doc() -> ExtractedDocument:
    return ExtractedDocument(
        source_path="test.txt",
        text="テスト用テキスト。",
        chunks=("テスト用テキスト。",),
        file_type="txt",
        used_ocr=False,
    )


@pytest.fixture
def mock_quality_report() -> QualityReport:
    return QualityReport(
        total_cards=1,
        passed_cards=1,
        critiqued_cards=0,
        removed_cards=0,
        improved_cards=0,
        split_cards=0,
        final_card_count=1,
    )


# ============================================================
# collect_files
# ============================================================


class TestCollectFiles:
    """Tests for collect_files."""

    def test_single_file(self, sample_txt: Path) -> None:
        """Single supported file returns a one-element list."""
        result = collect_files(sample_txt)
        assert result == [sample_txt]

    def test_directory_returns_supported_files(self, sample_dir: Path) -> None:
        """Directory returns only supported file types, sorted."""
        result = collect_files(sample_dir)
        suffixes = [f.suffix for f in result]
        assert ".csv" not in suffixes
        assert len(result) == 3  # .txt, .md, .pdf

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        """Nonexistent path raises ValueError."""
        with pytest.raises(ValueError, match="Path not found"):
            collect_files(tmp_path / "nonexistent")

    def test_unsupported_file_raises(self, tmp_path: Path) -> None:
        """Unsupported file type raises ValueError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file type"):
            collect_files(csv_file)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        """Directory with no supported files raises ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        (empty_dir / "data.csv").write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="No supported files"):
            collect_files(empty_dir)


# ============================================================
# resolve_output_path
# ============================================================


class TestResolveOutputPath:
    """Tests for resolve_output_path."""

    def test_explicit_output(self, tmp_path: Path) -> None:
        """Explicit -o path is returned as-is."""
        result = resolve_output_path(
            input_path=tmp_path / "test.txt",
            output=str(tmp_path / "out.tsv"),
            fmt="tsv",
            is_directory_input=False,
        )
        assert result == tmp_path / "out.tsv"

    def test_directory_input_default(self, tmp_path: Path) -> None:
        """Directory input without -o defaults to parent/output."""
        result = resolve_output_path(
            input_path=tmp_path / "docs",
            output=None,
            fmt="tsv",
            is_directory_input=True,
        )
        assert result == tmp_path / "output"

    def test_both_format_uses_parent(self, tmp_path: Path) -> None:
        """'both' format without -o uses input's parent directory."""
        result = resolve_output_path(
            input_path=tmp_path / "test.txt",
            output=None,
            fmt="both",
            is_directory_input=False,
        )
        assert result == tmp_path

    def test_single_file_default_suffix(self, tmp_path: Path) -> None:
        """Single file without -o uses input path with format suffix."""
        result = resolve_output_path(
            input_path=tmp_path / "test.txt",
            output=None,
            fmt="json",
            is_directory_input=False,
        )
        assert result == tmp_path / "test.json"


# ============================================================
# write_output
# ============================================================


class TestWriteOutput:
    """Tests for write_output."""

    def test_write_tsv(
        self,
        tmp_path: Path,
        mock_extraction_result: ExtractionResult,
    ) -> None:
        """TSV format writes a .tsv file."""
        output = tmp_path / "output.tsv"
        written = write_output(
            result=mock_extraction_result,
            output_path=output,
            fmt="tsv",
            source_stem="test",
            additional_tags=None,
        )
        assert len(written) == 1
        assert written[0].suffix == ".tsv"
        assert written[0].exists()

    def test_write_json(
        self,
        tmp_path: Path,
        mock_extraction_result: ExtractionResult,
    ) -> None:
        """JSON format writes a .json file."""
        output = tmp_path / "output.json"
        written = write_output(
            result=mock_extraction_result,
            output_path=output,
            fmt="json",
            source_stem="test",
            additional_tags=None,
        )
        assert len(written) == 1
        assert written[0].suffix == ".json"
        assert written[0].exists()

    def test_write_both(
        self,
        tmp_path: Path,
        mock_extraction_result: ExtractionResult,
    ) -> None:
        """'both' format writes both TSV and JSON files."""
        output_dir = tmp_path / "output"
        written = write_output(
            result=mock_extraction_result,
            output_path=output_dir,
            fmt="both",
            source_stem="test",
            additional_tags=None,
        )
        assert len(written) == 2
        suffixes = {f.suffix for f in written}
        assert suffixes == {".tsv", ".json"}


# ============================================================
# merge_quality_reports
# ============================================================


class TestMergeQualityReports:
    """Tests for merge_quality_reports."""

    def test_single_report(self, mock_quality_report: QualityReport) -> None:
        merged = merge_quality_reports([mock_quality_report])
        assert merged.total_cards == 1
        assert merged.passed_cards == 1

    def test_multiple_reports(self) -> None:
        r1 = QualityReport(
            total_cards=5,
            passed_cards=3,
            critiqued_cards=2,
            removed_cards=1,
            improved_cards=1,
            split_cards=0,
            final_card_count=4,
        )
        r2 = QualityReport(
            total_cards=10,
            passed_cards=8,
            critiqued_cards=2,
            removed_cards=0,
            improved_cards=2,
            split_cards=1,
            final_card_count=11,
        )
        merged = merge_quality_reports([r1, r2])
        assert merged.total_cards == 15
        assert merged.passed_cards == 11
        assert merged.critiqued_cards == 4
        assert merged.removed_cards == 1
        assert merged.improved_cards == 3
        assert merged.split_cards == 1
        assert merged.final_card_count == 15


# ============================================================
# process_file
# ============================================================


class TestProcessFile:
    """Tests for process_file."""

    @patch("pdf2anki.service.run_quality_pipeline")
    @patch("pdf2anki.service.extract_cards")
    @patch("pdf2anki.service.extract_with_cache")
    def test_with_quality_basic(
        self,
        mock_extract_text,
        mock_extract_cards,
        mock_quality_pipeline,
        sample_txt: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
        mock_cards: list[AnkiCard],
        mock_quality_report: QualityReport,
    ) -> None:
        """process_file with quality='basic' runs the quality pipeline."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )
        mock_quality_pipeline.return_value = (
            mock_cards,
            mock_quality_report,
            CostTracker(),
        )

        from pdf2anki.config import load_config

        config = load_config(None)
        result, report, tracker = process_file(
            file_path=sample_txt,
            config=config,
            cost_tracker=CostTracker(),
            quality="basic",
            focus_topics=None,
            additional_tags=None,
        )

        assert result.card_count >= 1
        assert report is not None
        mock_quality_pipeline.assert_called_once()

    @patch("pdf2anki.service.extract_cards")
    @patch("pdf2anki.service.extract_with_cache")
    def test_quality_off_skips_pipeline(
        self,
        mock_extract_text,
        mock_extract_cards,
        sample_txt: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ) -> None:
        """process_file with quality='off' skips the quality pipeline."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        from pdf2anki.config import load_config

        config = load_config(None)
        result, report, tracker = process_file(
            file_path=sample_txt,
            config=config,
            cost_tracker=CostTracker(),
            quality="off",
            focus_topics=None,
            additional_tags=None,
        )

        assert result.card_count >= 1
        assert report is None

    @patch("pdf2anki.service.collect_batch_results")
    @patch("pdf2anki.service.poll_batch")
    @patch("pdf2anki.service.submit_batch")
    @patch("pdf2anki.service.create_batch_requests")
    @patch("pdf2anki.service.extract_with_cache")
    def test_batch_mode(
        self,
        mock_extract_text,
        mock_create_requests,
        mock_submit,
        mock_poll,
        mock_collect,
        sample_txt: Path,
        mock_cards: list[AnkiCard],
    ) -> None:
        """process_file with batch=True uses batch API pipeline."""
        from pdf2anki.batch import BatchRequest, BatchResult
        from pdf2anki.config import load_config

        sections = (
            Section(
                id="section-0",
                heading="テスト",
                level=1,
                breadcrumb="テスト",
                text="# テスト\n\n本文。",
                page_range="",
                char_count=12,
            ),
        )
        doc = ExtractedDocument(
            source_path="test.txt",
            text="# テスト\n\n本文。",
            chunks=("# テスト\n\n本文。",),
            file_type="txt",
            used_ocr=False,
            sections=sections,
        )
        mock_extract_text.return_value = doc
        mock_create_requests.return_value = [
            BatchRequest(
                custom_id="section-0",
                model="claude-haiku-4-5-20251001",
                user_prompt="test",
                system_prompt="system",
                max_tokens=8192,
            ),
        ]
        mock_submit.return_value = "msgbatch_123"
        mock_poll.return_value = None
        mock_collect.return_value = [
            BatchResult(
                custom_id="section-0",
                cards=mock_cards,
                input_tokens=500,
                output_tokens=300,
                model="claude-haiku-4-5-20251001",
            ),
        ]

        config = load_config(None)
        result, report, tracker = process_file(
            file_path=sample_txt,
            config=config,
            cost_tracker=CostTracker(),
            quality="off",
            focus_topics=None,
            additional_tags=None,
            batch=True,
        )

        assert result.card_count >= 1
        mock_create_requests.assert_called_once()
        mock_submit.assert_called_once()


# ============================================================
# extract_with_cache
# ============================================================


class TestExtractWithCache:
    """Tests for extract_with_cache — cache integration with extract_text."""

    def test_cache_disabled_calls_extract(self, sample_txt: Path) -> None:
        """When cache_enabled=False, always calls extract_text."""
        from pdf2anki.config import AppConfig
        from pdf2anki.service import extract_with_cache

        config = AppConfig(cache_enabled=False)
        doc = extract_with_cache(sample_txt, config=config)
        assert doc.text  # extracted something
        assert doc.source_path == str(sample_txt)

    def test_cache_miss_extracts_and_caches(
        self, sample_txt: Path, tmp_path: Path
    ) -> None:
        """On cache miss, extract and write cache entry."""
        from pdf2anki.config import AppConfig
        from pdf2anki.service import extract_with_cache

        cache_dir = tmp_path / "cache"
        config = AppConfig(cache_enabled=True, cache_dir=str(cache_dir))
        doc = extract_with_cache(sample_txt, config=config)
        assert doc.text
        # Cache file should exist
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 1

    def test_cache_hit_skips_extraction(self, sample_txt: Path, tmp_path: Path) -> None:
        """On cache hit, return cached doc without calling extract_text again."""
        from pdf2anki.config import AppConfig
        from pdf2anki.service import extract_with_cache

        cache_dir = tmp_path / "cache"
        config = AppConfig(cache_enabled=True, cache_dir=str(cache_dir))

        # First call: extract and cache
        doc1 = extract_with_cache(sample_txt, config=config)

        # Second call: should hit cache
        with patch("pdf2anki.service.extract_text") as mock_et:
            doc2 = extract_with_cache(sample_txt, config=config)
            mock_et.assert_not_called()

        assert doc2.text == doc1.text

    def test_cache_invalidated_re_extracts(
        self, sample_txt: Path, tmp_path: Path
    ) -> None:
        """After content change (hash change), re-extract."""
        from pdf2anki.config import AppConfig
        from pdf2anki.service import extract_with_cache

        cache_dir = tmp_path / "cache"
        config = AppConfig(cache_enabled=True, cache_dir=str(cache_dir))

        doc1 = extract_with_cache(sample_txt, config=config)

        # Modify the file
        sample_txt.write_text("変更されたテキスト。", encoding="utf-8")

        doc2 = extract_with_cache(sample_txt, config=config)
        assert doc2.text != doc1.text
        assert "変更されたテキスト" in doc2.text
