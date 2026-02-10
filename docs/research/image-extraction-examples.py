"""PyMuPDF Image Extraction Examples for Feature 2: Image-Aware Card Generation

Research findings and practical examples for extracting images from PDFs
to support Claude Vision API integration.

References:
- PyMuPDF Images: https://pymupdf.readthedocs.io/en/latest/recipes-images.html
- PyMuPDF4LLM: https://github.com/pymupdf/pymupdf4llm
- Claude Vision API: https://platform.claude.com/docs/en/build-with-claude/vision
"""

from __future__ import annotations

import base64
from pathlib import Path

import pymupdf  # PyMuPDF


# =============================================================================
# 1. Image Detection API
# =============================================================================


def detect_images_on_page(page: pymupdf.Page) -> list[dict[str, any]]:
    """Detect all images on a PDF page.

    Returns:
        List of image dictionaries with keys:
        - xref: Cross-reference number (unique ID)
        - smask: Soft mask reference
        - width: Image width in pixels
        - height: Image height in pixels
        - bpc: Bits per component
        - colorspace: Color space name
        - alt: Alternative text (if any)
        - name: Image name (if any)
        - filter: Compression filter
        - bbox: Bounding box (x0, y0, x1, y1) in page coordinates
    """
    image_list = page.get_images(full=True)  # full=True returns detailed info
    images_with_bbox = []

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]  # Cross-reference number
        # Get bounding box for this image on the page
        bbox = page.get_image_bbox(xref)

        images_with_bbox.append({
            "xref": xref,
            "bbox": bbox,
            "index": img_index,
            "width": img_info[2],
            "height": img_info[3],
        })

    return images_with_bbox


def calculate_image_coverage(page: pymupdf.Page, threshold: float = 0.2) -> tuple[bool, float]:
    """Determine if a page has significant image content.

    Args:
        page: PDF page to analyze
        threshold: Minimum % of page area covered by images (default: 20%)

    Returns:
        Tuple of (has_significant_images, coverage_percentage)
    """
    images = detect_images_on_page(page)

    if not images:
        return False, 0.0

    # Calculate page area
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    # Calculate total image area
    total_image_area = 0.0
    for img in images:
        bbox = img["bbox"]
        img_width = bbox.x1 - bbox.x0
        img_height = bbox.y1 - bbox.y0
        total_image_area += img_width * img_height

    coverage = total_image_area / page_area if page_area > 0 else 0.0
    has_significant = coverage >= threshold

    return has_significant, coverage


# =============================================================================
# 2. Image Extraction Quality
# =============================================================================


def extract_image_high_quality(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    xref: int,
    output_path: Path,
    dpi: int = 150,
) -> dict[str, any]:
    """Extract a single image from PDF at specified DPI.

    Recommended DPI for Claude Vision API:
    - 150 DPI: Good balance of quality and file size (RECOMMENDED)
    - 300 DPI: High quality, but larger files
    - 72 DPI: Low quality, not recommended for text-heavy images

    Args:
        doc: PyMuPDF document
        page: PDF page containing the image
        xref: Image cross-reference number
        output_path: Where to save the extracted image
        dpi: Target DPI (dots per inch)

    Returns:
        Dict with extraction metadata:
        - path: Path to saved image
        - width: Image width in pixels
        - height: Image height in pixels
        - format: Image format (png, jpeg, etc.)
        - size_bytes: File size in bytes
    """
    # Method 1: Extract embedded image directly (fastest, preserves original format)
    img_dict = doc.extract_image(xref)
    image_bytes = img_dict["image"]
    image_ext = img_dict["ext"]  # jpeg, png, bmp, etc.

    # For Claude Vision API: ensure dimensions <= 1568px on long edge
    # and total pixels <= ~1.15 megapixels (1092x1092)
    pix = pymupdf.Pixmap(doc, xref)

    # Calculate if resizing is needed
    max_dimension = max(pix.width, pix.height)
    CLAUDE_MAX_DIMENSION = 1568
    CLAUDE_RECOMMENDED_MP = 1.15 * 1_000_000  # megapixels

    total_pixels = pix.width * pix.height

    if max_dimension > CLAUDE_MAX_DIMENSION or total_pixels > CLAUDE_RECOMMENDED_MP:
        # Calculate scale factor
        scale_for_dimension = CLAUDE_MAX_DIMENSION / max_dimension
        scale_for_pixels = (CLAUDE_RECOMMENDED_MP / total_pixels) ** 0.5
        scale = min(scale_for_dimension, scale_for_pixels)

        # Resize using DPI adjustment
        mat = pymupdf.Matrix(scale, scale)
        pix = pymupdf.Pixmap(doc, xref)
        pix = pymupdf.Pixmap(pix, mat)

    # Convert to RGB if needed (Claude Vision API prefers RGB)
    if pix.colorspace and pix.colorspace.name not in ("DeviceRGB", "DeviceGray"):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

    # Save as PNG (lossless) or JPEG (smaller file size)
    # PNG recommended for diagrams, charts, text-heavy images
    # JPEG recommended for photos
    output_ext = "png"  # Default to PNG for quality
    pix.save(str(output_path.with_suffix(f".{output_ext}")))

    file_size = output_path.with_suffix(f".{output_ext}").stat().st_size

    return {
        "path": str(output_path.with_suffix(f".{output_ext}")),
        "width": pix.width,
        "height": pix.height,
        "format": output_ext,
        "size_bytes": file_size,
    }


def extract_page_as_image(
    page: pymupdf.Page,
    output_path: Path,
    dpi: int = 150,
    colorspace: str = "rgb",
) -> dict[str, any]:
    """Render entire PDF page as an image (useful for complex layouts).

    Args:
        page: PDF page to render
        output_path: Where to save the page image
        dpi: Rendering DPI (default: 150)
        colorspace: "rgb" or "gray" (gray reduces file size by ~60%)

    Returns:
        Dict with rendering metadata
    """
    # Calculate scaling matrix for desired DPI
    # 72 DPI is the default, so scale = target_dpi / 72
    scale = dpi / 72.0
    mat = pymupdf.Matrix(scale, scale)

    # Render page to pixmap
    cs = pymupdf.csRGB if colorspace == "rgb" else pymupdf.csGRAY
    pix = page.get_pixmap(matrix=mat, colorspace=cs)

    # Claude Vision API: ensure dimensions are within limits
    max_dimension = max(pix.width, pix.height)
    CLAUDE_MAX_DIMENSION = 1568

    if max_dimension > CLAUDE_MAX_DIMENSION:
        # Scale down to fit
        scale_down = CLAUDE_MAX_DIMENSION / max_dimension
        mat_resize = pymupdf.Matrix(scale_down, scale_down)
        pix = page.get_pixmap(matrix=mat * mat_resize, colorspace=cs)

    # Save as PNG (recommended for page rendering)
    pix.save(str(output_path.with_suffix(".png")))

    file_size = output_path.with_suffix(".png").stat().st_size

    return {
        "path": str(output_path.with_suffix(".png")),
        "width": pix.width,
        "height": pix.height,
        "dpi": dpi,
        "colorspace": colorspace,
        "size_bytes": file_size,
    }


def convert_image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 string for inline embedding.

    Useful for pymupdf4llm's embed_images=True mode or direct API calls.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8")


# =============================================================================
# 3. Performance Considerations
# =============================================================================


def estimate_memory_usage(doc: pymupdf.Document) -> dict[str, any]:
    """Estimate memory usage for extracting all images from a PDF.

    Returns:
        Dict with memory estimates:
        - total_images: Total number of images
        - estimated_memory_mb: Estimated peak memory usage in MB
        - recommendation: "batch" or "on_demand"
    """
    total_images = 0
    total_pixels = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        total_images += len(images)

        for img_info in images:
            width = img_info[2]
            height = img_info[3]
            total_pixels += width * height

    # Rough estimate: 4 bytes per pixel (RGBA) + overhead
    estimated_memory_bytes = total_pixels * 4 * 1.5  # 1.5x for overhead
    estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)

    # Recommendation: batch process if < 500MB, on-demand if >= 500MB
    recommendation = "batch" if estimated_memory_mb < 500 else "on_demand"

    return {
        "total_images": total_images,
        "total_pixels": total_pixels,
        "estimated_memory_mb": round(estimated_memory_mb, 2),
        "recommendation": recommendation,
    }


def extract_images_on_demand(
    doc: pymupdf.Document,
    output_dir: Path,
    dpi: int = 150,
    page_filter: callable = None,
) -> list[dict[str, any]]:
    """Extract images page-by-page to minimize memory usage.

    Args:
        doc: PyMuPDF document
        output_dir: Directory to save extracted images
        dpi: Target DPI for extraction
        page_filter: Optional filter function(page_num, has_images, coverage) -> bool

    Returns:
        List of extracted image metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load page on-demand

        has_images, coverage = calculate_image_coverage(page)

        # Apply filter if provided
        if page_filter and not page_filter(page_num, has_images, coverage):
            continue

        if has_images:
            images = detect_images_on_page(page)

            for img in images:
                xref = img["xref"]
                output_path = output_dir / f"page_{page_num}_img_{img['index']}"

                try:
                    metadata = extract_image_high_quality(
                        doc, page, xref, output_path, dpi=dpi
                    )
                    metadata["page_num"] = page_num
                    metadata["coverage"] = coverage
                    extracted.append(metadata)
                except Exception as e:
                    print(f"Failed to extract image {xref} on page {page_num}: {e}")

        # Unload page to free memory
        page = None

    return extracted


# =============================================================================
# 4. Integration with pymupdf4llm
# =============================================================================


def extract_with_pymupdf4llm_images(
    pdf_path: Path,
    output_dir: Path,
    embed_images: bool = False,
) -> dict[str, any]:
    """Extract markdown with images using pymupdf4llm.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for markdown and images
        embed_images: If True, embed base64; if False, save images separately

    Returns:
        Dict with markdown text and image references
    """
    import pymupdf4llm

    output_dir.mkdir(parents=True, exist_ok=True)

    if embed_images:
        # Embed images as base64 (increases markdown size significantly)
        markdown_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            embed_images=True,
        )
        image_paths = []  # Images are embedded, no separate files
    else:
        # Save images separately and reference them in markdown
        image_output_dir = output_dir / "images"
        image_output_dir.mkdir(exist_ok=True)

        markdown_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            write_images=True,
            image_path=str(image_output_dir),
            image_format="png",  # png or jpg
            dpi=150,  # Optimal for Claude Vision API
        )

        # Collect saved image paths
        image_paths = list(image_output_dir.glob("*.png"))

    return {
        "markdown": markdown_text,
        "image_count": len(image_paths),
        "image_paths": [str(p) for p in image_paths],
        "embedded": embed_images,
    }


# =============================================================================
# Usage Example
# =============================================================================


def example_workflow(pdf_path: Path, output_dir: Path) -> None:
    """Example workflow for Feature 2 implementation."""

    doc = pymupdf.Document(str(pdf_path))

    # 1. Analyze memory requirements
    memory_info = estimate_memory_usage(doc)
    print(f"Memory estimate: {memory_info}")

    # 2. Detect image-heavy pages (>20% image coverage)
    image_heavy_pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        has_images, coverage = calculate_image_coverage(page, threshold=0.2)
        if has_images:
            image_heavy_pages.append((page_num, coverage))
            print(f"Page {page_num}: {coverage:.1%} image coverage")

    # 3. Extract images from image-heavy pages only
    def image_page_filter(page_num: int, has_images: bool, coverage: float) -> bool:
        return coverage >= 0.2  # Only extract if >20% coverage

    extracted = extract_images_on_demand(
        doc,
        output_dir / "images",
        dpi=150,  # Optimal for Claude Vision API
        page_filter=image_page_filter,
    )

    print(f"Extracted {len(extracted)} images from {len(image_heavy_pages)} pages")

    # 4. Alternative: Use pymupdf4llm for integrated extraction
    pymupdf4llm_result = extract_with_pymupdf4llm_images(
        pdf_path,
        output_dir / "pymupdf4llm",
        embed_images=False,  # Separate files recommended
    )

    print(f"pymupdf4llm extracted {pymupdf4llm_result['image_count']} images")


# =============================================================================
# Recommendations for Feature 2
# =============================================================================

"""
RECOMMENDATION SUMMARY:

1. IMAGE DETECTION:
   - Use page.get_images(full=True) to detect images
   - Calculate coverage = image_area / page_area
   - Threshold: 20% coverage = "image-heavy page"
   - Use page.get_image_bbox() for precise positioning

2. EXTRACTION QUALITY:
   - DPI: 150 (optimal for Claude Vision API)
   - Format: PNG for diagrams/text, JPEG for photos
   - Max dimension: 1568px (Claude Vision limit)
   - Max megapixels: ~1.15 MP (1092x1092px recommended)
   - Color: RGB preferred, grayscale for 60% size reduction

3. PERFORMANCE:
   - Memory estimate: ~6 bytes per pixel (with overhead)
   - Batch processing: < 500MB estimated memory
   - On-demand extraction: >= 500MB estimated memory
   - Use generators for large PDFs
   - Unload pages after processing to free memory

4. INTEGRATION WITH PYMUPDF4LLM:
   - Current code: pymupdf4llm.to_markdown() (text only)
   - Add write_images=True for separate image files (RECOMMENDED)
   - Add embed_images=True for base64 embedding (not recommended, large size)
   - Set dpi=150 for optimal Claude Vision API quality
   - Use image_format="png" for quality, "jpg" for smaller files

5. ARCHITECTURE APPROACH:
   - Detect image-heavy pages first (>20% coverage)
   - Extract images only from image-heavy pages (avoid overhead)
   - Store images separately (not embedded)
   - Pass image paths + markdown text to Claude Vision API
   - Use batch processing for documents with < 50 images
   - Use on-demand extraction for large documents (>50 images)

6. CLAUDE VISION API INTEGRATION:
   - Resize images to 1092x1092px for optimal performance
   - Use PNG format for text-heavy images (charts, diagrams)
   - Use JPEG format for photos
   - Include images in messages content array:
     {
       "role": "user",
       "content": [
         {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
         {"type": "text", "text": "Extract flashcards from this image..."}
       ]
     }

COMPATIBILITY NOTE:
- Current extract.py uses pymupdf4llm.to_markdown() without image support
- Add optional image_extraction=True parameter to extract_text()
- Return ExtractedDocument with new field: image_paths: tuple[str, ...]
- Maintain backward compatibility: default image_extraction=False
"""
