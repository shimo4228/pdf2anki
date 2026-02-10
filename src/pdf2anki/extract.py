"""Text extraction from PDF, TXT, and MD files.

Handles PDF→Markdown conversion via pymupdf4llm, plain text/Markdown
pass-through, OCR fallback for image-heavy PDFs, and preprocessing
(blank line normalization, control character removal).
Splits long documents into chunks under the token limit.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pymupdf4llm  # type: ignore[import-untyped]

from pdf2anki.image import ExtractedImage
from pdf2anki.section import Section, split_by_headings

# Approximate token limit for a single chunk (150K tokens ≈ 600K chars)
DEFAULT_TOKEN_LIMIT = 150_000
CHARS_PER_TOKEN = 4  # conservative estimate for mixed JP/EN
MIN_TEXT_LENGTH = 50  # threshold for OCR fallback

_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Control characters except \t (\x09), \n (\x0a), \r (\x0d)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 3+ consecutive blank lines (empty or whitespace-only lines)
_EXCESSIVE_BLANK_LINES_RE = re.compile(r"\n\s*\n(\s*\n)+")


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Result of text extraction from a single file."""

    source_path: str
    text: str
    chunks: tuple[str, ...]
    file_type: str
    used_ocr: bool
    sections: tuple[Section, ...] = ()
    images: tuple[ExtractedImage, ...] = ()


def preprocess_text(raw: str) -> str:
    """Normalize extracted text.

    - Remove control characters (keep newlines, tabs)
    - Collapse 3+ consecutive blank lines to 2
    - Strip leading/trailing whitespace
    """
    text = _CONTROL_CHAR_RE.sub("", raw)
    text = _EXCESSIVE_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def split_into_chunks(text: str, token_limit: int = DEFAULT_TOKEN_LIMIT) -> list[str]:
    """Split text into chunks under the token limit.

    Splits at paragraph boundaries (double newlines) when possible.
    Each chunk is at most token_limit * CHARS_PER_TOKEN characters.

    Args:
        text: The full text to split.
        token_limit: Max tokens per chunk (must be positive).

    Returns:
        List of text chunks.

    Raises:
        ValueError: If token_limit is not positive.
    """
    if token_limit <= 0:
        raise ValueError(f"token_limit must be positive, got {token_limit}")

    stripped = text.strip()
    if not stripped:
        return []

    char_limit = token_limit * CHARS_PER_TOKEN

    if len(stripped) <= char_limit:
        return [stripped]

    paragraphs = stripped.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        separator_len = 2 if current else 0

        if current and (current_len + separator_len + para_len) > char_limit:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        # Handle individual paragraph exceeding limit
        if para_len > char_limit and not current:
            for i in range(0, para_len, char_limit):
                chunks.append(para[i : i + char_limit])
        else:
            current.append(para)
            current_len += separator_len + para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF using pymupdf4llm."""
    result: str = pymupdf4llm.to_markdown(str(path))
    return result


def _extract_plain(path: Path) -> str:
    """Read a plain text or markdown file."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"File is not valid UTF-8: {path}. "
            "Please convert the file to UTF-8 encoding."
        ) from e


def extract_text(
    file_path: str | Path,
    *,
    ocr_enabled: bool = False,
    ocr_lang: str = "jpn+eng",
    token_limit: int = DEFAULT_TOKEN_LIMIT,
) -> ExtractedDocument:
    """Extract text from a PDF, TXT, or MD file.

    Args:
        file_path: Path to the input file.
        ocr_enabled: Whether to attempt OCR on image-heavy PDFs.
        ocr_lang: Language(s) for OCR (Tesseract format).
        token_limit: Max tokens per chunk for splitting.

    Returns:
        ExtractedDocument with extracted text and chunks.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If file type is unsupported.
    """
    path = Path(file_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    file_type = suffix.lstrip(".")
    used_ocr = False

    if suffix == ".pdf":
        raw_text = _extract_pdf(path)

        if ocr_enabled and len(raw_text.strip()) < MIN_TEXT_LENGTH:
            raw_text = _run_ocr(path, ocr_lang)
            used_ocr = True
    else:
        raw_text = _extract_plain(path)

    text = preprocess_text(raw_text)
    chunks = split_into_chunks(text, token_limit=token_limit)

    # Structure-aware sectioning for all file types
    sections = split_by_headings(text, document_title=path.stem)

    return ExtractedDocument(
        source_path=str(file_path),
        text=text,
        chunks=tuple(chunks),
        file_type=file_type,
        used_ocr=used_ocr,
        sections=tuple(sections),
    )


def _run_ocr(path: Path, lang: str) -> str:
    """Run OCR on a PDF file using ocrmypdf.

    Raises:
        ImportError: If ocrmypdf is not installed.
    """
    try:
        import ocrmypdf
    except ImportError as e:
        raise ImportError(
            "OCR requires the 'ocr' extra: pip install pdf2anki[ocr]"
        ) from e

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "ocr_output.pdf"
        ocrmypdf.ocr(str(path), str(tmp_path), language=lang, force_ocr=True)
        result: str = pymupdf4llm.to_markdown(str(tmp_path))
        return result
