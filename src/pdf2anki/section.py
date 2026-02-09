"""Structure-aware document sectioning for pdf2anki.

Splits Markdown text by heading boundaries (#, ##, ###) into Section objects
with breadcrumb context, page ranges, and automatic sub-splitting for
oversized sections. Includes Japanese heading detection fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Markdown heading pattern: # heading, ## heading, ### heading
_MD_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

# Japanese heading patterns (line-start anchored)
_JP_HEADING_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    # Level 1: chapter-level
    (re.compile(r"^(第\d+章.*)$", re.MULTILINE), 1),
    (re.compile(r"^(第[一二三四五六七八九十百]+章.*)$", re.MULTILINE), 1),
    (re.compile(r"^(序論|本論|結論)$", re.MULTILINE), 1),
    # Level 2: section-level
    (re.compile(r"^(\d+\.\s+.+)$", re.MULTILINE), 2),
    (re.compile(r"^(（\d+）.*)$", re.MULTILINE), 2),
    (re.compile(r"^([一二三四五六七八九十]、.*)$", re.MULTILINE), 2),
]


@dataclass(frozen=True, slots=True)
class Section:
    """A single section of a document with structural metadata."""

    id: str  # "section-0", "section-1-2"
    heading: str  # "序論", "第1章 論書名の意味"
    level: int  # 1=H1, 2=H2, 3=H3, 0=preamble
    breadcrumb: str  # "正理の海 > 本論 > 第1章"
    text: str  # Section body (including heading line)
    page_range: str  # "pp.3-18" or ""
    char_count: int  # len(text), precomputed


def _build_breadcrumb(heading_stack: list[str]) -> str:
    """Build a breadcrumb string from a heading stack.

    Args:
        heading_stack: List of headings from root to current level.

    Returns:
        Breadcrumb string joined with ' > '.
    """
    return " > ".join(heading_stack)


def _subsplit_section(section: Section, max_chars: int) -> list[Section]:
    """Sub-split an oversized section at paragraph boundaries.

    Args:
        section: The oversized Section to split.
        max_chars: Maximum characters per sub-section.

    Returns:
        List of sub-sections with updated IDs and char_counts.
    """
    if section.char_count <= max_chars:
        return [section]

    paragraphs = section.text.split("\n\n")
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        separator_len = 2 if current_parts else 0

        if current_parts and (current_len + separator_len + para_len) > max_chars:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0

        # Handle single paragraph exceeding limit
        if para_len > max_chars and not current_parts:
            for i in range(0, para_len, max_chars):
                chunks.append(para[i : i + max_chars])
        else:
            current_parts.append(para)
            current_len += separator_len + para_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    if len(chunks) <= 1:
        return [section]

    result: list[Section] = []
    for i, chunk_text in enumerate(chunks):
        result.append(
            Section(
                id=f"{section.id}-{i}",
                heading=section.heading if i == 0 else f"{section.heading} (cont.)",
                level=section.level,
                breadcrumb=section.breadcrumb,
                text=chunk_text,
                page_range=section.page_range,
                char_count=len(chunk_text),
            )
        )

    return result


def _detect_japanese_headings(text: str) -> list[tuple[int, str, int]]:
    """Detect Japanese-style headings in plain text.

    Patterns detected:
    - 第\\d+章, 第[一二三...]章
    - 序論|本論|結論
    - \\d+\\.\\s+
    - （\\d+）
    - [一二三...]、

    Args:
        text: Plain text to scan for headings.

    Returns:
        List of (line_position, heading_text, level) tuples sorted by position.
    """
    found: dict[int, tuple[int, str, int]] = {}

    for pattern, level in _JP_HEADING_PATTERNS:
        for match in pattern.finditer(text):
            pos = match.start()
            heading_text = match.group(1).strip()
            if pos not in found:
                found[pos] = (pos, heading_text, level)

    return sorted(found.values(), key=lambda x: x[0])


def split_by_headings(
    markdown_text: str,
    *,
    max_chars: int = 30_000,
    document_title: str = "",
) -> list[Section]:
    """Split markdown text by heading boundaries into Section objects.

    Splits at #/##/### headings. Builds breadcrumb stack from heading
    hierarchy. Oversized sections are sub-split at paragraph boundaries.
    Fallback: Japanese heading detection -> single Section.

    Args:
        markdown_text: Full document text (Markdown format).
        max_chars: Maximum characters per section before sub-splitting.
        document_title: Document title for breadcrumb root.

    Returns:
        List of Section objects, ordered by document position.
    """
    stripped = markdown_text.strip()
    if not stripped:
        return []

    # Try markdown headings first
    heading_matches = list(_MD_HEADING_RE.finditer(stripped))

    if heading_matches:
        return _split_by_md_headings(
            stripped,
            heading_matches,
            max_chars=max_chars,
            document_title=document_title,
        )

    # Fallback: try Japanese heading detection
    jp_headings = _detect_japanese_headings(stripped)
    if jp_headings:
        return _split_by_jp_headings(
            stripped, jp_headings, max_chars=max_chars, document_title=document_title
        )

    # No headings found: return as single preamble section
    stack = [document_title] if document_title else []
    breadcrumb = _build_breadcrumb(stack) if stack else ""
    section = Section(
        id="section-0",
        heading="",
        level=0,
        breadcrumb=breadcrumb,
        text=stripped,
        page_range="",
        char_count=len(stripped),
    )
    return _subsplit_section(section, max_chars)


def _split_by_md_headings(
    text: str,
    matches: list[re.Match[str]],
    *,
    max_chars: int,
    document_title: str,
) -> list[Section]:
    """Split text using markdown heading positions."""
    sections: list[Section] = []
    section_idx = 0

    # heading_stack[level] = heading text (1-indexed)
    heading_stack: dict[int, str] = {}

    # Handle preamble (text before first heading)
    first_pos = matches[0].start()
    preamble_text = text[:first_pos].strip()
    if preamble_text:
        stack = [document_title] if document_title else []
        breadcrumb = _build_breadcrumb(stack) if stack else ""
        preamble = Section(
            id=f"section-{section_idx}",
            heading="",
            level=0,
            breadcrumb=breadcrumb,
            text=preamble_text,
            page_range="",
            char_count=len(preamble_text),
        )
        sections.extend(_subsplit_section(preamble, max_chars))
        section_idx += 1

    for i, match in enumerate(matches):
        hashes = match.group(1)
        heading_text = match.group(2).strip()
        level = len(hashes)

        # Determine section end position
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[match.start() : end_pos].strip()

        # Update heading stack: clear deeper levels when a higher level appears
        keys_to_remove = [k for k in heading_stack if k >= level]
        for k in keys_to_remove:
            del heading_stack[k]
        heading_stack[level] = heading_text

        # Build breadcrumb from stack
        stack_list: list[str] = []
        if document_title:
            stack_list.append(document_title)
        for lvl in sorted(heading_stack.keys()):
            stack_list.append(heading_stack[lvl])
        breadcrumb = _build_breadcrumb(stack_list)

        section = Section(
            id=f"section-{section_idx}",
            heading=heading_text,
            level=level,
            breadcrumb=breadcrumb,
            text=section_text,
            page_range="",
            char_count=len(section_text),
        )
        sections.extend(_subsplit_section(section, max_chars))
        section_idx += 1

    return sections


def _split_by_jp_headings(
    text: str,
    headings: list[tuple[int, str, int]],
    *,
    max_chars: int,
    document_title: str,
) -> list[Section]:
    """Split text using Japanese heading positions."""
    sections: list[Section] = []
    section_idx = 0

    heading_stack: dict[int, str] = {}

    # Handle preamble
    first_pos = headings[0][0]
    preamble_text = text[:first_pos].strip()
    if preamble_text:
        stack = [document_title] if document_title else []
        breadcrumb = _build_breadcrumb(stack) if stack else ""
        preamble = Section(
            id=f"section-{section_idx}",
            heading="",
            level=0,
            breadcrumb=breadcrumb,
            text=preamble_text,
            page_range="",
            char_count=len(preamble_text),
        )
        sections.extend(_subsplit_section(preamble, max_chars))
        section_idx += 1

    for i, (pos, heading_text, level) in enumerate(headings):
        end_pos = headings[i + 1][0] if i + 1 < len(headings) else len(text)

        section_text = text[pos:end_pos].strip()

        keys_to_remove = [k for k in heading_stack if k >= level]
        for k in keys_to_remove:
            del heading_stack[k]
        heading_stack[level] = heading_text

        stack_list: list[str] = []
        if document_title:
            stack_list.append(document_title)
        for lvl in sorted(heading_stack.keys()):
            stack_list.append(heading_stack[lvl])
        breadcrumb = _build_breadcrumb(stack_list)

        section = Section(
            id=f"section-{section_idx}",
            heading=heading_text,
            level=level,
            breadcrumb=breadcrumb,
            text=section_text,
            page_range="",
            char_count=len(section_text),
        )
        sections.extend(_subsplit_section(section, max_chars))
        section_idx += 1

    return sections


def extract_page_ranges(page_chunks: list[dict[str, Any]]) -> dict[str, str]:
    """Extract page ranges from pymupdf4llm page_chunks toc_items.

    Args:
        page_chunks: List of dicts from pymupdf4llm with toc_items.

    Returns:
        Dict mapping heading text to page range string (e.g., "pp.3-18").
    """
    if not page_chunks:
        return {}

    # Collect (title -> list of pages) from toc_items
    title_pages: dict[str, list[int]] = {}

    for chunk in page_chunks:
        toc_items = chunk.get("toc_items", [])
        page_num = chunk.get("metadata", {}).get("page", 0)

        for item in toc_items:
            title = item.get("title", "")
            if title:
                if title not in title_pages:
                    title_pages[title] = []
                toc_page = item.get("page", page_num)
                title_pages[title].append(toc_page)

    # Build page range strings
    result: dict[str, str] = {}
    for title, pages in title_pages.items():
        if pages:
            min_page = min(pages)
            max_page = max(pages)
            if min_page == max_page:
                result[title] = f"p.{min_page}"
            else:
                result[title] = f"pp.{min_page}-{max_page}"

    return result
