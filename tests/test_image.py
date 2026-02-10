"""Tests for pdf2anki.image — image detection, extraction, and utilities."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from pdf2anki.image import (
    VISION_MAX_DIMENSION,
    VISION_MAX_MEGAPIXELS,
    ExtractedImage,
    PageImageInfo,
    detect_page_images,
    estimate_image_tokens,
    extract_page_images,
    image_to_base64,
)

# ---------------------------------------------------------------------------
# ExtractedImage dataclass
# ---------------------------------------------------------------------------


class TestExtractedImage:
    """ExtractedImage is a frozen dataclass."""

    def test_creation(self) -> None:
        img = ExtractedImage(
            page_num=0,
            index=0,
            width=800,
            height=600,
            bbox=(0.0, 0.0, 400.0, 300.0),
            image_bytes=b"\x89PNG",
            media_type="image/png",
            coverage=0.35,
        )
        assert img.page_num == 0
        assert img.width == 800
        assert img.height == 600
        assert img.media_type == "image/png"
        assert img.coverage == 0.35

    def test_frozen(self) -> None:
        img = ExtractedImage(
            page_num=0,
            index=0,
            width=100,
            height=100,
            bbox=(0.0, 0.0, 100.0, 100.0),
            image_bytes=b"x",
            media_type="image/png",
            coverage=0.5,
        )
        with pytest.raises(AttributeError):
            img.width = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PageImageInfo dataclass
# ---------------------------------------------------------------------------


class TestPageImageInfo:
    """PageImageInfo is a frozen dataclass."""

    def test_creation_no_images(self) -> None:
        info = PageImageInfo(
            page_num=0,
            has_significant_images=False,
            coverage=0.0,
            images=(),
        )
        assert info.has_significant_images is False
        assert info.images == ()

    def test_creation_with_images(self) -> None:
        img = ExtractedImage(
            page_num=0,
            index=0,
            width=400,
            height=300,
            bbox=(0.0, 0.0, 400.0, 300.0),
            image_bytes=b"\x89PNG",
            media_type="image/png",
            coverage=0.25,
        )
        info = PageImageInfo(
            page_num=0,
            has_significant_images=True,
            coverage=0.25,
            images=(img,),
        )
        assert info.has_significant_images is True
        assert len(info.images) == 1


# ---------------------------------------------------------------------------
# detect_page_images
# ---------------------------------------------------------------------------


def _make_mock_page(
    *,
    images: list[tuple[int, int, int]],  # (xref, width, height)
    page_width: float = 595.0,  # A4 width in points
    page_height: float = 842.0,  # A4 height in points
    bbox_positions: list[tuple[float, float, float, float]] | None = None,
) -> MagicMock:
    """Create a mock pymupdf Page with images.

    Args:
        images: List of (xref, width, height) tuples for get_images(full=True).
        page_width: Page width in points.
        page_height: Page height in points.
        bbox_positions: Override bboxes. If None, generates from (0,0).
    """
    page = MagicMock()

    # get_images(full=True) returns tuples:
    # (xref, smask, width, height, bpc, colorspace, alt, name, filter)
    image_list = []
    for xref, w, h in images:
        image_list.append((xref, 0, w, h, 8, "DeviceRGB", "", "", ""))

    page.get_images.return_value = image_list

    # page.rect → mock Rect
    rect = MagicMock()
    rect.width = page_width
    rect.height = page_height
    page.rect = rect

    # get_image_bbox for each xref
    if bbox_positions is None:
        # Generate default bboxes based on image dimensions
        bboxes = {}
        for _i, (xref, w, h) in enumerate(images):
            # Scale image dimensions to page coordinates (rough approximation)
            scale = min(page_width / max(w, 1), page_height / max(h, 1), 1.0)
            bboxes[xref] = MagicMock()
            bboxes[xref].x0 = 0.0
            bboxes[xref].y0 = 0.0
            bboxes[xref].x1 = w * scale
            bboxes[xref].y1 = h * scale
    else:
        bboxes = {}
        for i, (xref, _, _) in enumerate(images):
            bboxes[xref] = MagicMock()
            bboxes[xref].x0 = bbox_positions[i][0]
            bboxes[xref].y0 = bbox_positions[i][1]
            bboxes[xref].x1 = bbox_positions[i][2]
            bboxes[xref].y1 = bbox_positions[i][3]

    def mock_get_image_bbox(img_item: tuple[int, ...]) -> MagicMock:
        xref = img_item[0]
        return bboxes.get(xref, MagicMock(x0=0, y0=0, x1=0, y1=0))

    page.get_image_bbox.side_effect = mock_get_image_bbox

    return page


class TestDetectPageImages:
    """detect_page_images returns PageImageInfo for a page."""

    def test_no_images(self) -> None:
        page = _make_mock_page(images=[])
        info = detect_page_images(page)
        assert info.has_significant_images is False
        assert info.coverage == 0.0
        assert info.images == ()

    def test_single_large_image(self) -> None:
        # Image covers 50% of page
        page = _make_mock_page(
            images=[(1, 595, 421)],
            bbox_positions=[(0.0, 0.0, 595.0, 421.0)],
        )
        info = detect_page_images(page)
        assert info.has_significant_images is True
        assert info.coverage > 0.2

    def test_small_image_below_threshold(self) -> None:
        # Image covers ~5% of page
        page = _make_mock_page(
            images=[(1, 100, 50)],
            bbox_positions=[(0.0, 0.0, 100.0, 50.0)],
            page_width=595.0,
            page_height=842.0,
        )
        info = detect_page_images(page, coverage_threshold=0.20)
        # 100*50 / (595*842) = 0.01 → below threshold
        assert info.has_significant_images is False

    def test_custom_threshold(self) -> None:
        # Image covers ~10% of page
        page = _make_mock_page(
            images=[(1, 200, 250)],
            bbox_positions=[(0.0, 0.0, 200.0, 250.0)],
        )
        info = detect_page_images(page, coverage_threshold=0.05)
        # 200*250 / (595*842) = 0.0998 → above 5% threshold
        assert info.has_significant_images is True

    def test_multiple_images(self) -> None:
        page = _make_mock_page(
            images=[(1, 200, 200), (2, 300, 200)],
            bbox_positions=[
                (0.0, 0.0, 200.0, 200.0),
                (200.0, 0.0, 500.0, 200.0),
            ],
        )
        info = detect_page_images(page)
        assert info.page_num == 0  # default


# ---------------------------------------------------------------------------
# extract_page_images
# ---------------------------------------------------------------------------


def _make_mock_doc(
    pages: list[MagicMock],
    *,
    image_bytes_map: dict[int, bytes] | None = None,
) -> MagicMock:
    """Create a mock pymupdf Document with pages.

    Args:
        pages: List of mock pages.
        image_bytes_map: xref → image bytes mapping for extract_image.
    """
    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=len(pages))
    doc.load_page.side_effect = lambda i: pages[i]

    # extract_image(xref) → {"image": bytes, "ext": "png", ...}
    if image_bytes_map is None:
        image_bytes_map = {}

    def mock_extract_image(xref: int) -> dict[str, bytes | str | int]:
        return {
            "image": image_bytes_map.get(xref, b"\x89PNG\r\n\x1a\n"),
            "ext": "png",
            "width": 400,
            "height": 300,
        }

    doc.extract_image.side_effect = mock_extract_image
    return doc


class TestExtractPageImages:
    """extract_page_images processes all pages and returns image-heavy ones."""

    def test_empty_doc(self) -> None:
        doc = _make_mock_doc([])
        result = extract_page_images(doc)
        assert result == []

    def test_no_image_pages(self) -> None:
        page = _make_mock_page(images=[])
        doc = _make_mock_doc([page])
        result = extract_page_images(doc)
        assert result == []

    @patch("pdf2anki.image._extract_and_resize")
    def test_extracts_from_image_heavy_page(self, mock_resize: MagicMock) -> None:
        mock_resize.return_value = (b"\x89PNG", "image/png", 500, 600)
        page = _make_mock_page(
            images=[(1, 500, 600)],
            bbox_positions=[(0.0, 0.0, 500.0, 600.0)],
        )
        doc = _make_mock_doc([page])
        result = extract_page_images(doc)
        assert len(result) == 1
        assert result[0].has_significant_images is True
        assert len(result[0].images) >= 1

    @patch("pdf2anki.image._extract_and_resize")
    def test_skips_text_only_pages(self, mock_resize: MagicMock) -> None:
        mock_resize.return_value = (b"\x89PNG", "image/png", 500, 600)
        text_page = _make_mock_page(images=[])
        image_page = _make_mock_page(
            images=[(1, 500, 600)],
            bbox_positions=[(0.0, 0.0, 500.0, 600.0)],
        )
        doc = _make_mock_doc([text_page, image_page])
        result = extract_page_images(doc)
        assert len(result) == 1
        assert result[0].page_num == 1

    def test_max_images_per_page(self) -> None:
        page = _make_mock_page(
            images=[(i, 200, 200) for i in range(10)],
            bbox_positions=[(0.0, i * 84.0, 200.0, (i + 1) * 84.0) for i in range(10)],
        )
        doc = _make_mock_doc([page])
        result = extract_page_images(doc, max_images_per_page=3)
        if result:
            assert len(result[0].images) <= 3

    @patch("pdf2anki.image._extract_and_resize")
    def test_custom_coverage_threshold(self, mock_resize: MagicMock) -> None:
        mock_resize.return_value = (b"\x89PNG", "image/png", 100, 100)
        # Small image: 100*100 / (595*842) ≈ 2%
        page = _make_mock_page(
            images=[(1, 100, 100)],
            bbox_positions=[(0.0, 0.0, 100.0, 100.0)],
        )
        doc = _make_mock_doc([page])

        result_high = extract_page_images(doc, coverage_threshold=0.20)
        assert len(result_high) == 0

        result_low = extract_page_images(doc, coverage_threshold=0.01)
        assert len(result_low) == 1


# ---------------------------------------------------------------------------
# image_to_base64
# ---------------------------------------------------------------------------


class TestImageToBase64:
    """image_to_base64 encodes image bytes to base64 string."""

    def test_basic_encode(self) -> None:
        img = ExtractedImage(
            page_num=0,
            index=0,
            width=100,
            height=100,
            bbox=(0.0, 0.0, 100.0, 100.0),
            image_bytes=b"\x89PNG\r\n\x1a\n",
            media_type="image/png",
            coverage=0.5,
        )
        result = image_to_base64(img)
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert decoded == b"\x89PNG\r\n\x1a\n"

    def test_empty_bytes(self) -> None:
        img = ExtractedImage(
            page_num=0,
            index=0,
            width=0,
            height=0,
            bbox=(0.0, 0.0, 0.0, 0.0),
            image_bytes=b"",
            media_type="image/png",
            coverage=0.0,
        )
        result = image_to_base64(img)
        assert result == ""


# ---------------------------------------------------------------------------
# estimate_image_tokens
# ---------------------------------------------------------------------------


class TestEstimateImageTokens:
    """estimate_image_tokens calculates Claude Vision API token count."""

    def test_standard_image(self) -> None:
        # 1092x1092 ≈ 1590 tokens
        tokens = estimate_image_tokens(1092, 1092)
        assert tokens == (1092 * 1092) // 750

    def test_small_image(self) -> None:
        tokens = estimate_image_tokens(100, 100)
        assert tokens == (100 * 100) // 750

    def test_zero_dimensions(self) -> None:
        tokens = estimate_image_tokens(0, 0)
        assert tokens == 0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Vision API constants are correct."""

    def test_max_dimension(self) -> None:
        assert VISION_MAX_DIMENSION == 1568

    def test_max_megapixels(self) -> None:
        assert VISION_MAX_MEGAPIXELS == 1_150_000
