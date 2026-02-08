"""Tests for pdf2anki.extract module."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from pdf2anki.extract import (
    DEFAULT_TOKEN_LIMIT,
    ExtractedDocument,
    extract_text,
    preprocess_text,
    split_into_chunks,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# preprocess_text tests
# ---------------------------------------------------------------------------


class TestPreprocessText:
    """Tests for the preprocess_text utility."""

    def test_strips_leading_trailing_whitespace(self) -> None:
        result = preprocess_text("  hello world  \n")
        assert result == "hello world"

    def test_removes_control_characters(self) -> None:
        result = preprocess_text("hello\x00world\x07test\x08end")
        assert result == "helloworldtestend"

    def test_preserves_newlines_and_tabs(self) -> None:
        result = preprocess_text("line1\nline2\tindented")
        assert result == "line1\nline2\tindented"

    def test_collapses_excessive_blank_lines(self) -> None:
        text = "paragraph1\n\n\n\n\nparagraph2"
        result = preprocess_text(text)
        assert result == "paragraph1\n\nparagraph2"

    def test_preserves_double_blank_lines(self) -> None:
        text = "paragraph1\n\nparagraph2"
        result = preprocess_text(text)
        assert result == "paragraph1\n\nparagraph2"

    def test_empty_string(self) -> None:
        result = preprocess_text("")
        assert result == ""

    def test_japanese_text_preserved(self) -> None:
        text = "日本語テスト\n\n次の段落"
        result = preprocess_text(text)
        assert result == "日本語テスト\n\n次の段落"

    def test_windows_line_endings(self) -> None:
        text = "line1\r\nline2\r\n\r\n\r\n\r\nline3"
        result = preprocess_text(text)
        # \r should be preserved (it's not a control char we remove)
        # Excessive blank lines should still be collapsed
        assert "line1" in result
        assert "line3" in result


# ---------------------------------------------------------------------------
# split_into_chunks tests
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    """Tests for the split_into_chunks utility."""

    def test_short_text_single_chunk(self) -> None:
        text = "short text"
        chunks = split_into_chunks(text, token_limit=1000)
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_splits_at_paragraph_boundaries(self) -> None:
        para1 = "A" * 100
        para2 = "B" * 100
        text = f"{para1}\n\n{para2}"
        # token_limit=30 → char_limit=120, each para fits alone but not together
        chunks = split_into_chunks(text, token_limit=30)
        assert len(chunks) == 2
        assert chunks[0].strip() == para1
        assert chunks[1].strip() == para2

    def test_empty_text_returns_empty_list(self) -> None:
        chunks = split_into_chunks("")
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        chunks = split_into_chunks("   \n\n   ")
        assert chunks == []

    def test_single_paragraph_exceeding_limit(self) -> None:
        long_text = "X" * 1000
        chunks = split_into_chunks(long_text, token_limit=10)
        combined = "".join(chunks)
        assert combined == long_text
        assert len(chunks) >= 1

    def test_preserves_all_content(self) -> None:
        paragraphs = [f"Paragraph {i}: " + "x" * 50 for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = split_into_chunks(text, token_limit=50)
        combined = "\n\n".join(chunks)
        for para in paragraphs:
            assert para in combined

    def test_default_token_limit(self) -> None:
        chunks = split_into_chunks("short")
        assert len(chunks) == 1

    def test_zero_token_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="token_limit must be positive"):
            split_into_chunks("text", token_limit=0)

    def test_negative_token_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="token_limit must be positive"):
            split_into_chunks("text", token_limit=-5)


# ---------------------------------------------------------------------------
# extract_text tests - TXT
# ---------------------------------------------------------------------------


class TestExtractTextTxt:
    """Tests for extracting text from .txt files."""

    def test_extract_txt_basic(self) -> None:
        result = extract_text(FIXTURES_DIR / "sample.txt")
        assert isinstance(result, ExtractedDocument)
        assert result.file_type == "txt"
        assert result.used_ocr is False
        assert "ニューラルネットワーク" in result.text
        assert len(result.chunks) >= 1

    def test_extract_txt_source_path(self) -> None:
        path = FIXTURES_DIR / "sample.txt"
        result = extract_text(path)
        assert result.source_path == str(path)

    def test_extract_txt_preprocessed(self) -> None:
        result = extract_text(FIXTURES_DIR / "sample.txt")
        assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", result.text)

    def test_extract_txt_non_utf8(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.txt"
        bad_file.write_bytes(b"\xff\xfe invalid utf-8")
        with pytest.raises(ValueError, match="not valid UTF-8"):
            extract_text(bad_file)


# ---------------------------------------------------------------------------
# extract_text tests - MD
# ---------------------------------------------------------------------------


class TestExtractTextMd:
    """Tests for extracting text from .md files."""

    def test_extract_md_basic(self) -> None:
        result = extract_text(FIXTURES_DIR / "sample.md")
        assert isinstance(result, ExtractedDocument)
        assert result.file_type == "md"
        assert result.used_ocr is False
        assert "機械学習入門" in result.text
        assert len(result.chunks) >= 1


# ---------------------------------------------------------------------------
# extract_text tests - PDF
# ---------------------------------------------------------------------------


class TestExtractTextPdf:
    """Tests for extracting text from .pdf files."""

    def test_extract_pdf_basic(self, sample_pdf: Path) -> None:
        result = extract_text(sample_pdf)
        assert isinstance(result, ExtractedDocument)
        assert result.file_type == "pdf"
        assert result.used_ocr is False
        assert len(result.text) > 0
        assert len(result.chunks) >= 1

    def test_extract_pdf_source_path(self, sample_pdf: Path) -> None:
        result = extract_text(sample_pdf)
        assert result.source_path == str(sample_pdf)


# ---------------------------------------------------------------------------
# extract_text tests - OCR fallback
# ---------------------------------------------------------------------------


class TestExtractTextOcr:
    """Tests for OCR fallback behavior."""

    def test_ocr_triggered_when_text_short(self, sample_pdf: Path) -> None:
        with patch("pdf2anki.extract._extract_pdf", return_value=""):
            with patch(
                "pdf2anki.extract._run_ocr", return_value="OCR result text here"
            ) as mock_ocr:
                result = extract_text(sample_pdf, ocr_enabled=True)
                mock_ocr.assert_called_once()
                assert result.used_ocr is True

    def test_ocr_not_triggered_when_disabled(self, sample_pdf: Path) -> None:
        with patch("pdf2anki.extract._extract_pdf", return_value=""):
            with patch("pdf2anki.extract._run_ocr") as mock_ocr:
                result = extract_text(sample_pdf, ocr_enabled=False)
                mock_ocr.assert_not_called()
                assert result.used_ocr is False

    def test_ocr_not_triggered_when_text_sufficient(self, sample_pdf: Path) -> None:
        long_text = "A" * 100
        with patch("pdf2anki.extract._extract_pdf", return_value=long_text):
            with patch("pdf2anki.extract._run_ocr") as mock_ocr:
                result = extract_text(sample_pdf, ocr_enabled=True)
                mock_ocr.assert_not_called()
                assert result.used_ocr is False


# ---------------------------------------------------------------------------
# extract_text tests - Error cases
# ---------------------------------------------------------------------------


class TestExtractTextErrors:
    """Tests for error handling in extract_text."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_text(tmp_path / "nonexistent.txt")

    def test_unsupported_file_type(self, tmp_path: Path) -> None:
        unsupported = tmp_path / "test.csv"
        unsupported.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(unsupported)

    def test_string_path_accepted(self) -> None:
        result = extract_text(str(FIXTURES_DIR / "sample.txt"))
        assert isinstance(result, ExtractedDocument)

    def test_directory_raises_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_text(tmp_path)


# ---------------------------------------------------------------------------
# extract_text tests - Chunking integration
# ---------------------------------------------------------------------------


class TestExtractTextChunking:
    """Tests for chunking within extract_text."""

    def test_small_file_single_chunk(self) -> None:
        result = extract_text(FIXTURES_DIR / "sample.txt")
        assert len(result.chunks) == 1

    def test_custom_token_limit(self) -> None:
        result = extract_text(FIXTURES_DIR / "sample.txt", token_limit=20)
        assert len(result.chunks) > 1

    def test_chunks_are_tuple(self) -> None:
        result = extract_text(FIXTURES_DIR / "sample.txt")
        assert isinstance(result.chunks, tuple)


# ---------------------------------------------------------------------------
# ExtractedDocument immutability
# ---------------------------------------------------------------------------


class TestExtractedDocumentImmutability:
    """Tests for ExtractedDocument frozen dataclass."""

    def test_cannot_mutate_fields(self) -> None:
        doc = ExtractedDocument(
            source_path="/test.txt",
            text="hello",
            chunks=("hello",),
            file_type="txt",
            used_ocr=False,
        )
        with pytest.raises(AttributeError):
            doc.text = "modified"  # type: ignore[misc]

    def test_chunks_are_immutable_tuple(self) -> None:
        doc = ExtractedDocument(
            source_path="/test.txt",
            text="hello",
            chunks=("hello",),
            file_type="txt",
            used_ocr=False,
        )
        with pytest.raises(TypeError):
            doc.chunks[0] = "modified"  # type: ignore[index]
