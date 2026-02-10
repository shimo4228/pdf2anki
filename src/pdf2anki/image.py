"""Image detection and extraction from PDF pages.

Detects image-heavy pages, extracts images for Claude Vision API,
and provides utility functions for base64 encoding and token estimation.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import pymupdf  # type: ignore[import-untyped]

# Claude Vision API constraints
VISION_MAX_DIMENSION = 1568  # px (long edge)
VISION_MAX_MEGAPIXELS = 1_150_000  # 1.15 MP in pixels
VISION_RECOMMENDED_DPI = 150

# Type alias for detected image tuple
_Detected = tuple[int, int, int, int, tuple[float, float, float, float], float]


@dataclass(frozen=True, slots=True)
class ExtractedImage:
    """A single image extracted from a PDF page."""

    page_num: int
    index: int
    width: int
    height: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    image_bytes: bytes
    media_type: str  # "image/png" | "image/jpeg"
    coverage: float  # fraction of page area covered


@dataclass(frozen=True, slots=True)
class PageImageInfo:
    """Image analysis result for a single page."""

    page_num: int
    has_significant_images: bool
    coverage: float
    images: tuple[ExtractedImage, ...]


def detect_page_images(
    page: Any,
    *,
    page_num: int = 0,
    coverage_threshold: float = 0.20,
) -> PageImageInfo:
    """Detect images on a PDF page and calculate coverage.

    Args:
        page: A pymupdf.Page object.
        page_num: Page number (0-indexed).
        coverage_threshold: Minimum coverage fraction to flag as significant.

    Returns:
        PageImageInfo with detected images (no extraction yet).
    """
    image_list = page.get_images(full=True)

    if not image_list:
        return PageImageInfo(
            page_num=page_num,
            has_significant_images=False,
            coverage=0.0,
            images=(),
        )

    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    if page_area <= 0:
        return PageImageInfo(
            page_num=page_num,
            has_significant_images=False,
            coverage=0.0,
            images=(),
        )

    total_image_area = 0.0
    detected: list[_Detected] = []

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        img_width = img_info[2]
        img_height = img_info[3]

        bbox_obj = page.get_image_bbox(img_info)
        x0, y0 = bbox_obj.x0, bbox_obj.y0
        x1, y1 = bbox_obj.x1, bbox_obj.y1

        bbox_area = (x1 - x0) * (y1 - y0)
        total_image_area += max(bbox_area, 0.0)

        detected.append(
            (xref, img_index, img_width, img_height,
             (x0, y0, x1, y1), bbox_area),
        )

    coverage = total_image_area / page_area
    has_significant = coverage >= coverage_threshold

    # Build lightweight image info (no bytes yet)
    images: list[ExtractedImage] = []
    for _xref, idx, w, h, bbox, area in detected:
        images.append(
            ExtractedImage(
                page_num=page_num,
                index=idx,
                width=w,
                height=h,
                bbox=bbox,
                image_bytes=b"",  # populated by extract_page_images
                media_type="image/png",
                coverage=area / page_area if page_area > 0 else 0.0,
            )
        )

    return PageImageInfo(
        page_num=page_num,
        has_significant_images=has_significant,
        coverage=coverage,
        images=tuple(images),
    )


def _extract_and_resize(
    doc: Any,
    xref: int,
) -> tuple[bytes, str, int, int]:
    """Extract image from document and resize for Vision API limits.

    Returns:
        (image_bytes, media_type, width, height)
    """
    pix = pymupdf.Pixmap(doc, xref)

    # Resize if exceeding Vision API limits
    max_dim = max(pix.width, pix.height)
    total_pixels = pix.width * pix.height

    if max_dim > VISION_MAX_DIMENSION or total_pixels > VISION_MAX_MEGAPIXELS:
        scale_dim = (
            VISION_MAX_DIMENSION / max_dim
            if max_dim > VISION_MAX_DIMENSION
            else 1.0
        )
        scale_px = (
            (VISION_MAX_MEGAPIXELS / total_pixels) ** 0.5
            if total_pixels > VISION_MAX_MEGAPIXELS
            else 1.0
        )
        scale = min(scale_dim, scale_px)
        mat = pymupdf.Matrix(scale, scale)
        pix = pymupdf.Pixmap(pix, mat)

    # Convert CMYK / other colorspaces to RGB
    if (pix.n > 3 and pix.alpha == 0) or pix.n > 4:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

    image_bytes: bytes = pix.tobytes("png")
    return image_bytes, "image/png", pix.width, pix.height


def extract_page_images(
    doc: Any,
    *,
    coverage_threshold: float = 0.20,
    max_images_per_page: int = 5,
) -> list[PageImageInfo]:
    """Analyze all pages and extract images from image-heavy pages.

    Args:
        doc: A pymupdf.Document.
        coverage_threshold: Min coverage fraction for page to be image-heavy.
        max_images_per_page: Max images to extract per page.

    Returns:
        List of PageImageInfo for pages with significant images.
    """
    results: list[PageImageInfo] = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        info = detect_page_images(
            page,
            page_num=page_num,
            coverage_threshold=coverage_threshold,
        )

        if not info.has_significant_images:
            continue

        # Extract actual image bytes for significant pages
        extracted: list[ExtractedImage] = []
        for img in info.images[:max_images_per_page]:
            xref = page.get_images(full=True)[img.index][0]
            try:
                img_bytes, media_type, w, h = _extract_and_resize(
                    doc, xref,
                )
                extracted.append(
                    ExtractedImage(
                        page_num=page_num,
                        index=img.index,
                        width=w,
                        height=h,
                        bbox=img.bbox,
                        image_bytes=img_bytes,
                        media_type=media_type,
                        coverage=img.coverage,
                    )
                )
            except Exception:  # noqa: BLE001
                # Skip corrupted or unsupported images
                continue

        results.append(
            PageImageInfo(
                page_num=page_num,
                has_significant_images=True,
                coverage=info.coverage,
                images=tuple(extracted),
            )
        )

    return results


def image_to_base64(image: ExtractedImage) -> str:
    """Encode image bytes to base64 string for Vision API.

    Returns empty string for empty image bytes.
    """
    if not image.image_bytes:
        return ""
    return base64.b64encode(image.image_bytes).decode("ascii")


def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate Claude Vision API input tokens for an image.

    Formula: tokens = (width * height) / 750
    Reference: 1092x1092 ~ 1,590 tokens
    """
    return (width * height) // 750
