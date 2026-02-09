"""Tests for pdf2anki CLI (main.py).

Covers:
- convert subcommand (single file, directory, all output formats)
- preview subcommand (dry-run text extraction)
- CLI options (--quality, --format, --tags, --focus, etc.)
- Error handling (missing file, bad config, unsupported format)
- Directory batch processing
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pdf2anki.batch import BatchResult
from pdf2anki.cost import CostTracker
from pdf2anki.extract import ExtractedDocument
from pdf2anki.main import _merge_quality_reports, _parse_csv_option
from pdf2anki.quality import QualityReport
from pdf2anki.schemas import AnkiCard, BloomLevel, CardType, ExtractionResult
from pdf2anki.section import Section


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    p = tmp_path / "test.txt"
    p.write_text(
        "ニューラルネットワークの基礎\n活性化関数はReLUが代表的。",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("文書A: テスト用テキスト。", encoding="utf-8")
    (d / "b.md").write_text("# 文書B\nマークダウン。", encoding="utf-8")
    (d / "ignore.csv").write_text("col1,col2\n1,2", encoding="utf-8")
    return d


@pytest.fixture
def mock_cards() -> list[AnkiCard]:
    return [
        AnkiCard(
            front="ReLUの数式は何ですか？",
            back="f(x) = max(0, x)",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI::活性化関数"],
        ),
        AnkiCard(
            front="{{c1::勾配降下法}}は損失関数を最小化する。",
            back="",
            card_type=CardType.CLOZE,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI::最適化"],
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
def mock_quality_report() -> QualityReport:
    return QualityReport(
        total_cards=2,
        passed_cards=2,
        critiqued_cards=0,
        removed_cards=0,
        improved_cards=0,
        split_cards=0,
        final_card_count=2,
    )


@pytest.fixture
def mock_extracted_doc() -> ExtractedDocument:
    return ExtractedDocument(
        source_path="test.txt",
        text="ニューラルネットワークの基礎\n活性化関数はReLUが代表的。",
        chunks=("ニューラルネットワークの基礎\n活性化関数はReLUが代表的。",),
        file_type="txt",
        used_ocr=False,
    )


def _get_app():
    """Lazily import the app to avoid import errors during RED phase."""
    from pdf2anki.main import app
    return app


# ============================================================
# convert subcommand: basic functionality
# ============================================================


class TestConvertBasic:
    """Basic convert subcommand tests."""

    @patch("pdf2anki.main.run_quality_pipeline")
    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_convert_txt_to_tsv(
        self,
        mock_extract_text,
        mock_extract_cards,
        mock_quality_pipeline,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
        mock_cards: list[AnkiCard],
        mock_quality_report: QualityReport,
    ):
        """Convert a text file to TSV (default format)."""
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

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output)],
        )

        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "#separator:tab" in content
        assert "ReLU" in content

    @patch("pdf2anki.main.run_quality_pipeline")
    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_convert_to_json(
        self,
        mock_extract_text,
        mock_extract_cards,
        mock_quality_pipeline,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
        mock_cards: list[AnkiCard],
        mock_quality_report: QualityReport,
    ):
        """Convert to JSON format."""
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

        output = tmp_path / "output.json"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert '"source_file"' in content
        assert '"_meta"' in content

    @patch("pdf2anki.main.run_quality_pipeline")
    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_convert_to_both(
        self,
        mock_extract_text,
        mock_extract_cards,
        mock_quality_pipeline,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
        mock_cards: list[AnkiCard],
        mock_quality_report: QualityReport,
    ):
        """Convert to both TSV and JSON."""
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

        output_dir = tmp_path / "output"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output_dir),
                "--format",
                "both",
            ],
        )

        assert result.exit_code == 0
        tsv_files = list(output_dir.glob("*.tsv"))
        json_files = list(output_dir.glob("*.json"))
        assert len(tsv_files) == 1
        assert len(json_files) == 1

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_convert_default_output_path(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
        mock_cards: list[AnkiCard],
    ):
        """When no -o, output goes to same dir as input."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        result = runner.invoke(_get_app(), ["convert", str(sample_txt)])

        assert result.exit_code == 0
        expected_out = sample_txt.with_suffix(".tsv")
        assert expected_out.exists()


# ============================================================
# convert subcommand: options
# ============================================================


class TestConvertOptions:
    """Tests for convert command options."""

    @patch("pdf2anki.main.run_quality_pipeline")
    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_quality_full(
        self,
        mock_extract_text,
        mock_extract_cards,
        mock_quality_pipeline,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
        mock_cards: list[AnkiCard],
        mock_quality_report: QualityReport,
    ):
        """--quality full runs the quality pipeline."""
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

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "full"],
        )

        assert result.exit_code == 0
        mock_quality_pipeline.assert_called_once()

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_quality_off_skips_pipeline(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """--quality off skips quality pipeline entirely."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "off"],
        )

        assert result.exit_code == 0

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_max_cards_option(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """--max-cards passes to config."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--max-cards",
                "10",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_extract_cards.call_args
        config = (
            call_kwargs.kwargs.get("config")
            or call_kwargs[1].get("config")
        )
        assert config.cards_max_cards == 10

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_tags_option(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """--tags passes additional tags to extract_cards."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--tags",
                "AI::基礎,test",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_extract_cards.call_args
        additional_tags = (
            call_kwargs.kwargs.get("additional_tags")
            or call_kwargs[1].get("additional_tags")
        )
        assert "AI::基礎" in additional_tags
        assert "test" in additional_tags

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_focus_option(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """--focus passes focus topics to extract_cards."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--focus",
                "CNN,RNN",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_extract_cards.call_args
        focus_topics = (
            call_kwargs.kwargs.get("focus_topics")
            or call_kwargs[1].get("focus_topics")
        )
        assert "CNN" in focus_topics
        assert "RNN" in focus_topics

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_bloom_filter_option(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """--bloom-filter passes to extract_cards."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--bloom-filter",
                "remember,understand",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_extract_cards.call_args
        bloom_filter = (
            call_kwargs.kwargs.get("bloom_filter")
            or call_kwargs[1].get("bloom_filter")
        )
        assert "remember" in bloom_filter
        assert "understand" in bloom_filter

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_budget_limit_option(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """--budget-limit passes to config."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--budget-limit",
                "0.50",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_extract_cards.call_args
        config = (
            call_kwargs.kwargs.get("config")
            or call_kwargs[1].get("config")
        )
        assert config.cost_budget_limit == 0.50

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_ocr_option(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """--ocr enables OCR extraction."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--ocr",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_extract_text.call_args
        assert call_kwargs.kwargs.get("ocr_enabled") is True

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_model_option(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """--model overrides the default model in config."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--model",
                "claude-haiku-4-5-20251001",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_extract_cards.call_args
        config = (
            call_kwargs.kwargs.get("config")
            or call_kwargs[1].get("config")
        )
        assert config.model == "claude-haiku-4-5-20251001"

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_verbose_option(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """--verbose enables debug logging."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--verbose",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0


# ============================================================
# convert subcommand: directory batch processing
# ============================================================


class TestConvertDirectory:
    """Tests for directory batch processing."""

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_convert_directory(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_dir: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """Convert all supported files in a directory."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output_dir = tmp_path / "output"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_dir),
                "-o",
                str(output_dir),
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0
        # Should process .txt and .md, skip .csv
        assert mock_extract_text.call_count == 2


# ============================================================
# convert subcommand: error handling
# ============================================================


class TestConvertErrors:
    """Tests for error handling in convert command."""

    def test_missing_file(self, runner: CliRunner):
        """Error when input file doesn't exist."""
        result = runner.invoke(_get_app(), ["convert", "/nonexistent/file.txt"])
        assert result.exit_code != 0

    def test_unsupported_file_type(self, runner: CliRunner, tmp_path: Path):
        """Error for unsupported file types."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c", encoding="utf-8")
        result = runner.invoke(_get_app(), ["convert", str(csv_file)])
        assert result.exit_code != 0

    def test_bad_config_path(self, runner: CliRunner, sample_txt: Path):
        """Error when config file doesn't exist."""
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "--config", "/nonexistent/config.yaml"],
        )
        assert result.exit_code != 0


# ============================================================
# preview subcommand
# ============================================================


class TestPreview:
    """Tests for preview (dry-run) subcommand."""

    @patch("pdf2anki.main.extract_text")
    def test_preview_shows_text(
        self,
        mock_extract_text,
        runner: CliRunner,
        sample_txt: Path,
        mock_extracted_doc: ExtractedDocument,
    ):
        """Preview displays extracted text and metadata."""
        mock_extract_text.return_value = mock_extracted_doc

        result = runner.invoke(_get_app(), ["preview", str(sample_txt)])

        assert result.exit_code == 0
        assert "test.txt" in result.output or "txt" in result.output

    @patch("pdf2anki.main.extract_text")
    def test_preview_shows_chunks(
        self,
        mock_extract_text,
        runner: CliRunner,
        sample_txt: Path,
    ):
        """Preview displays chunk information."""
        doc = ExtractedDocument(
            source_path="test.txt",
            text="chunk1 text\n\nchunk2 text",
            chunks=("chunk1 text", "chunk2 text"),
            file_type="txt",
            used_ocr=False,
        )
        mock_extract_text.return_value = doc

        result = runner.invoke(_get_app(), ["preview", str(sample_txt)])

        assert result.exit_code == 0
        assert "2" in result.output  # 2 chunks shown somewhere

    def test_preview_missing_file(self, runner: CliRunner):
        """Preview errors on missing file."""
        result = runner.invoke(_get_app(), ["preview", "/nonexistent/file.txt"])
        assert result.exit_code != 0

    @patch("pdf2anki.main.extract_text")
    def test_preview_with_ocr(
        self,
        mock_extract_text,
        runner: CliRunner,
        sample_txt: Path,
        mock_extracted_doc: ExtractedDocument,
    ):
        """Preview with OCR flag."""
        mock_extract_text.return_value = mock_extracted_doc

        result = runner.invoke(_get_app(), ["preview", str(sample_txt), "--ocr"])

        assert result.exit_code == 0
        call_kwargs = mock_extract_text.call_args
        assert call_kwargs.kwargs.get("ocr_enabled") is True


# ============================================================
# Output content verification
# ============================================================


class TestOutputContent:
    """Verify output file content correctness."""

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_tsv_contains_bloom_tags(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """TSV output includes bloom:: tags."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "off"],
        )

        assert result.exit_code == 0
        content = output.read_text(encoding="utf-8")
        assert "bloom::remember" in content

    @patch("pdf2anki.main.run_quality_pipeline")
    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_convert_shows_summary(  # noqa: PLR0913
        self,
        mock_extract_text,
        mock_extract_cards,
        mock_quality_pipeline,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
        mock_cards: list[AnkiCard],
        mock_quality_report: QualityReport,
    ):
        """Convert prints summary with card count and cost."""
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

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output)],
        )

        assert result.exit_code == 0
        # Should show card count in output
        assert "2" in result.output


# ============================================================
# Helper function unit tests
# ============================================================


class TestParseCsvOption:
    """Tests for _parse_csv_option helper."""

    def test_none_returns_none(self):
        assert _parse_csv_option(None) is None

    def test_single_value(self):
        assert _parse_csv_option("foo") == ["foo"]

    def test_multiple_values(self):
        assert _parse_csv_option("a,b,c") == ["a", "b", "c"]

    def test_strips_whitespace(self):
        assert _parse_csv_option(" a , b , c ") == ["a", "b", "c"]


class TestMergeQualityReports:
    """Tests for _merge_quality_reports helper."""

    def test_single_report(self, mock_quality_report: QualityReport):
        merged = _merge_quality_reports([mock_quality_report])
        assert merged.total_cards == 2
        assert merged.passed_cards == 2

    def test_multiple_reports(self):
        r1 = QualityReport(
            total_cards=5, passed_cards=3, critiqued_cards=2,
            removed_cards=1, improved_cards=1, split_cards=0, final_card_count=4,
        )
        r2 = QualityReport(
            total_cards=10, passed_cards=8, critiqued_cards=2,
            removed_cards=0, improved_cards=2, split_cards=1, final_card_count=11,
        )
        merged = _merge_quality_reports([r1, r2])
        assert merged.total_cards == 15
        assert merged.passed_cards == 11
        assert merged.critiqued_cards == 4
        assert merged.removed_cards == 1
        assert merged.improved_cards == 3
        assert merged.split_cards == 1
        assert merged.final_card_count == 15


class TestProcessFileErrorHandling:
    """Tests for error handling during file processing."""

    @patch(
        "pdf2anki.main.extract_cards",
        side_effect=RuntimeError("Budget exceeded"),
    )
    @patch("pdf2anki.main.extract_text")
    def test_convert_continues_on_error(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_dir: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
    ):
        """Convert continues processing when a file fails."""
        mock_extract_text.return_value = mock_extracted_doc

        output_dir = tmp_path / "output"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_dir),
                "-o",
                str(output_dir),
                "--quality",
                "off",
            ],
        )

        # Should not crash, prints errors and continues
        assert result.exit_code == 0
        assert "Error processing" in result.output


# ============================================================
# _process_file: section-aware processing (Phase 2)
# ============================================================


class TestProcessFileSections:
    """Test that _process_file passes sections to extract_cards."""

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_sections_passed_to_extract_cards(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extraction_result: ExtractionResult,
    ):
        """When doc has sections, _process_file should pass them to extract_cards."""
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
        doc_with_sections = ExtractedDocument(
            source_path="test.txt",
            text="# テスト\n\n本文。",
            chunks=("# テスト\n\n本文。",),
            file_type="txt",
            used_ocr=False,
            sections=sections,
        )
        mock_extract_text.return_value = doc_with_sections
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "off"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_extract_cards.call_args
        passed_sections = (
            call_kwargs.kwargs.get("sections")
            or call_kwargs[1].get("sections")
        )
        assert passed_sections is not None
        assert len(passed_sections) == 1

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_no_sections_uses_chunks(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """When doc has no sections, _process_file should not pass sections."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "off"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_extract_cards.call_args
        passed_sections = (
            call_kwargs.kwargs.get("sections")
            or call_kwargs[1].get("sections")
        )
        # No sections passed (None or not present)
        assert passed_sections is None


# ============================================================
# --batch CLI flag (Phase 3)
# ============================================================


class TestBatchFlag:
    """Test --batch flag for batch API processing."""

    @patch("pdf2anki.main.collect_batch_results")
    @patch("pdf2anki.main.poll_batch")
    @patch("pdf2anki.main.submit_batch")
    @patch("pdf2anki.main.create_batch_requests")
    @patch("pdf2anki.main.extract_text")
    def test_batch_flag_uses_batch_pipeline(
        self,
        mock_extract_text,
        mock_create_requests,
        mock_submit,
        mock_poll,
        mock_collect,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_cards: list[AnkiCard],
    ):
        """--batch flag should use batch API pipeline instead of standard."""
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
        from pdf2anki.batch import BatchRequest

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

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--batch",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0
        mock_create_requests.assert_called_once()
        mock_submit.assert_called_once()
        mock_poll.assert_called_once()
        mock_collect.assert_called_once()

    @patch("pdf2anki.main.extract_cards")
    @patch("pdf2anki.main.extract_text")
    def test_batch_without_sections_falls_back(
        self,
        mock_extract_text,
        mock_extract_cards,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
        mock_extracted_doc: ExtractedDocument,
        mock_extraction_result: ExtractionResult,
    ):
        """--batch without sections should fall back to standard processing."""
        mock_extract_text.return_value = mock_extracted_doc
        mock_extract_cards.return_value = (
            mock_extraction_result,
            CostTracker(),
        )

        output = tmp_path / "out.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--batch",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0
        # Falls back to standard extract_cards
        mock_extract_cards.assert_called_once()
