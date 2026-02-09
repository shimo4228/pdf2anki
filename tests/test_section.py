"""Tests for pdf2anki.section - Structure-aware document sectioning.

TDD RED phase: Tests written before implementation.

Tests cover:
- Section dataclass: creation, immutability, fields
- split_by_headings(): H1/H2/H3 splitting, breadcrumb, preamble, subsplit
- _detect_japanese_headings(): all pattern types
- _subsplit_section(): paragraph boundary splitting
- _build_breadcrumb(): heading stack joining
- extract_page_ranges(): toc_items page range mapping
- Edge cases: empty text, no headings, mixed levels
"""

from __future__ import annotations

import pytest

from pdf2anki.section import (
    Section,
    _build_breadcrumb,
    _detect_japanese_headings,
    _subsplit_section,
    extract_page_ranges,
    split_by_headings,
)

# ============================================================
# Section dataclass tests
# ============================================================


class TestSectionDataclass:
    """Test Section frozen dataclass creation and immutability."""

    def test_section_creation(self) -> None:
        """Section should be created with all required fields."""
        section = Section(
            id="section-0",
            heading="序論",
            level=1,
            breadcrumb="正理の海 > 序論",
            text="# 序論\n\nここに本文が入る。",
            page_range="pp.1-5",
            char_count=18,
        )
        assert section.id == "section-0"
        assert section.heading == "序論"
        assert section.level == 1
        assert section.breadcrumb == "正理の海 > 序論"
        assert section.text == "# 序論\n\nここに本文が入る。"
        assert section.page_range == "pp.1-5"
        assert section.char_count == 18

    def test_section_immutability(self) -> None:
        """Section should be immutable (frozen=True)."""
        section = Section(
            id="section-0",
            heading="テスト",
            level=1,
            breadcrumb="テスト",
            text="本文",
            page_range="",
            char_count=2,
        )
        with pytest.raises(AttributeError):
            section.heading = "変更"  # type: ignore[misc]

    def test_section_slots(self) -> None:
        """Section should use __slots__ for memory efficiency."""
        section = Section(
            id="section-0",
            heading="テスト",
            level=1,
            breadcrumb="テスト",
            text="本文",
            page_range="",
            char_count=2,
        )
        with pytest.raises((AttributeError, TypeError)):
            section.extra_field = "not allowed"  # type: ignore[attr-defined]

    def test_section_char_count_matches_text(self) -> None:
        """char_count should be set to len(text) by convention."""
        text = "# 見出し\n\n本文テキスト"
        section = Section(
            id="section-0",
            heading="見出し",
            level=1,
            breadcrumb="見出し",
            text=text,
            page_range="",
            char_count=len(text),
        )
        assert section.char_count == len(text)


# ============================================================
# _build_breadcrumb tests
# ============================================================


class TestBuildBreadcrumb:
    """Test breadcrumb string generation."""

    def test_single_level(self) -> None:
        """Single heading should produce simple breadcrumb."""
        assert _build_breadcrumb(["序論"]) == "序論"

    def test_multi_level(self) -> None:
        """Multiple headings should be joined with ' > '."""
        result = _build_breadcrumb(["正理の海", "本論", "第1章"])
        assert result == "正理の海 > 本論 > 第1章"

    def test_empty_stack(self) -> None:
        """Empty heading stack should produce empty string."""
        assert _build_breadcrumb([]) == ""

    def test_two_levels(self) -> None:
        """Two-level breadcrumb."""
        result = _build_breadcrumb(["ドキュメント", "セクション1"])
        assert result == "ドキュメント > セクション1"


# ============================================================
# split_by_headings tests - Basic H1/H2/H3
# ============================================================


class TestSplitByHeadingsBasic:
    """Test basic heading-based splitting."""

    def test_single_h1_section(self) -> None:
        """Single H1 section should produce one Section."""
        md = "# 第1章 概要\n\n本文テキストが入ります。"
        sections = split_by_headings(md)
        assert len(sections) == 1
        assert sections[0].heading == "第1章 概要"
        assert sections[0].level == 1
        assert "本文テキスト" in sections[0].text

    def test_two_h1_sections(self) -> None:
        """Two H1 sections should produce two Sections."""
        md = "# 第1章\n\n第1章の内容。\n\n# 第2章\n\n第2章の内容。"
        sections = split_by_headings(md)
        assert len(sections) == 2
        assert sections[0].heading == "第1章"
        assert sections[1].heading == "第2章"

    def test_h2_under_h1(self) -> None:
        """H2 under H1 should produce separate sections with correct hierarchy."""
        md = (
            "# 本論\n\n概要テキスト。\n\n"
            "## 第1節\n\n第1節の内容。\n\n"
            "## 第2節\n\n第2節の内容。"
        )
        sections = split_by_headings(md)
        assert len(sections) >= 3
        h2_sections = [s for s in sections if s.level == 2]
        assert len(h2_sections) == 2

    def test_h3_under_h2(self) -> None:
        """H3 under H2 should produce correct levels."""
        md = (
            "# 本論\n\n概要。\n\n"
            "## 第1節\n\nテキスト。\n\n"
            "### 第1項\n\n項目の内容。"
        )
        sections = split_by_headings(md)
        h3_sections = [s for s in sections if s.level == 3]
        assert len(h3_sections) == 1
        assert h3_sections[0].heading == "第1項"


# ============================================================
# split_by_headings tests - Breadcrumb generation
# ============================================================


class TestSplitByHeadingsBreadcrumb:
    """Test breadcrumb context generation."""

    def test_h1_breadcrumb(self) -> None:
        """H1 section should have heading as breadcrumb."""
        md = "# 序論\n\n序論の内容。"
        sections = split_by_headings(md)
        assert sections[0].breadcrumb == "序論"

    def test_h1_breadcrumb_with_document_title(self) -> None:
        """H1 section with document_title should include title in breadcrumb."""
        md = "# 序論\n\n序論の内容。"
        sections = split_by_headings(md, document_title="正理の海")
        assert "正理の海" in sections[0].breadcrumb
        assert "序論" in sections[0].breadcrumb

    def test_h2_breadcrumb_includes_h1(self) -> None:
        """H2 breadcrumb should include parent H1."""
        md = "# 本論\n\n概要。\n\n## 第1章\n\n内容。"
        sections = split_by_headings(md)
        h2_section = [s for s in sections if s.level == 2][0]
        assert "本論" in h2_section.breadcrumb
        assert "第1章" in h2_section.breadcrumb

    def test_h3_breadcrumb_includes_h1_and_h2(self) -> None:
        """H3 breadcrumb should include parent H1 and H2."""
        md = "# 本論\n\n概要。\n\n## 第1章\n\n概要。\n\n### 第1節\n\n内容。"
        sections = split_by_headings(md)
        h3_section = [s for s in sections if s.level == 3][0]
        assert "本論" in h3_section.breadcrumb
        assert "第1章" in h3_section.breadcrumb
        assert "第1節" in h3_section.breadcrumb

    def test_h2_resets_on_new_h1(self) -> None:
        """New H1 should reset the breadcrumb stack."""
        md = (
            "# 本論\n\n概要。\n\n"
            "## 第1章\n\n内容。\n\n"
            "# 結論\n\n結論の内容。\n\n"
            "## まとめ\n\nまとめの内容。"
        )
        sections = split_by_headings(md)
        summary = [s for s in sections if s.heading == "まとめ"][0]
        assert "結論" in summary.breadcrumb
        assert "本論" not in summary.breadcrumb


# ============================================================
# split_by_headings tests - Preamble
# ============================================================


class TestSplitByHeadingsPreamble:
    """Test handling of text before first heading."""

    def test_preamble_becomes_section(self) -> None:
        """Text before first heading should become a level-0 preamble section."""
        md = "これはプリアンブルです。\n\n# 第1章\n\n本文。"
        sections = split_by_headings(md)
        preamble = sections[0]
        assert preamble.level == 0
        assert "プリアンブル" in preamble.text

    def test_no_preamble_when_starts_with_heading(self) -> None:
        """No preamble section when text starts with a heading."""
        md = "# 第1章\n\n本文。"
        sections = split_by_headings(md)
        assert sections[0].level == 1
        assert sections[0].heading == "第1章"


# ============================================================
# split_by_headings tests - Section IDs
# ============================================================


class TestSplitByHeadingsIds:
    """Test section ID generation."""

    def test_sequential_ids(self) -> None:
        """Sections should have sequential IDs."""
        md = "# A\n\nText A.\n\n# B\n\nText B.\n\n# C\n\nText C."
        sections = split_by_headings(md)
        ids = [s.id for s in sections]
        assert ids == ["section-0", "section-1", "section-2"]

    def test_preamble_id(self) -> None:
        """Preamble should have section-0 id."""
        md = "Preamble text.\n\n# First\n\nContent."
        sections = split_by_headings(md)
        assert sections[0].id == "section-0"
        assert sections[1].id == "section-1"


# ============================================================
# split_by_headings tests - Oversized section sub-splitting
# ============================================================


class TestSplitByHeadingsSubsplit:
    """Test sub-splitting of oversized sections."""

    def test_oversized_section_is_subsplit(self) -> None:
        """Section exceeding max_chars should be sub-split."""
        # Create a section with >100 chars
        long_body = "\n\n".join([f"段落{i}。" + "あ" * 50 for i in range(5)])
        md = f"# 長いセクション\n\n{long_body}"
        sections = split_by_headings(md, max_chars=100)
        assert len(sections) > 1

    def test_subsplit_preserves_heading_in_first(self) -> None:
        """First sub-section should contain the original heading."""
        long_body = "\n\n".join(["段落テキスト。" + "あ" * 50 for _ in range(5)])
        md = f"# 長いセクション\n\n{long_body}"
        sections = split_by_headings(md, max_chars=100)
        assert sections[0].heading == "長いセクション"

    def test_subsplit_ids_have_sub_suffix(self) -> None:
        """Sub-split sections should have IDs like 'section-0-0', 'section-0-1'."""
        long_body = "\n\n".join(["段落テキスト。" + "あ" * 50 for _ in range(5)])
        md = f"# 長いセクション\n\n{long_body}"
        sections = split_by_headings(md, max_chars=100)
        # At least the subsplit IDs should contain '-'
        assert any("-" in s.id.replace("section-", "", 1) for s in sections)


# ============================================================
# split_by_headings tests - Edge cases
# ============================================================


class TestSplitByHeadingsEdgeCases:
    """Test edge cases for split_by_headings."""

    def test_empty_text(self) -> None:
        """Empty text should return empty list."""
        sections = split_by_headings("")
        assert sections == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only text should return empty list."""
        sections = split_by_headings("   \n\n   ")
        assert sections == []

    def test_no_headings_fallback(self) -> None:
        """Text without headings should fall back to single Section."""
        md = "これは見出しのない文章です。\n\n段落が続きます。"
        sections = split_by_headings(md)
        assert len(sections) >= 1
        assert sections[0].level == 0

    def test_heading_only_no_body(self) -> None:
        """Heading with no body should still produce a Section."""
        md = "# 空のセクション"
        sections = split_by_headings(md)
        assert len(sections) == 1
        assert sections[0].heading == "空のセクション"

    def test_mixed_heading_levels(self) -> None:
        """Mixed heading levels should all be parsed correctly."""
        md = (
            "# H1\n\n本文。\n\n"
            "### H3直接\n\n内容。\n\n"
            "## H2\n\n内容。"
        )
        sections = split_by_headings(md)
        levels = [s.level for s in sections]
        assert 1 in levels
        assert 2 in levels
        assert 3 in levels

    def test_char_count_is_set(self) -> None:
        """Each Section.char_count should equal len(Section.text)."""
        md = "# セクション\n\n本文テキスト。"
        sections = split_by_headings(md)
        for section in sections:
            assert section.char_count == len(section.text)

    def test_document_title_only(self) -> None:
        """document_title with no headings should appear in breadcrumb."""
        md = "見出しなしの本文。"
        sections = split_by_headings(md, document_title="テスト文書")
        assert len(sections) == 1
        assert "テスト文書" in sections[0].breadcrumb


# ============================================================
# _detect_japanese_headings tests
# ============================================================


class TestDetectJapaneseHeadings:
    """Test Japanese heading pattern detection."""

    def test_arabic_chapter_pattern(self) -> None:
        """Detect '第1章', '第2章' etc."""
        text = "概要文。\n第1章 論書名の意味\n内容。\n第2章 結論\n内容。"
        headings = _detect_japanese_headings(text)
        assert len(headings) >= 2
        assert any("第1章" in h[1] for h in headings)
        assert any("第2章" in h[1] for h in headings)

    def test_kanji_chapter_pattern(self) -> None:
        """Detect '第一章', '第二章' etc."""
        text = "概要。\n第一章 序論\n内容。\n第二章 本論\n内容。"
        headings = _detect_japanese_headings(text)
        assert len(headings) >= 2

    def test_section_labels(self) -> None:
        """Detect '序論', '本論', '結論'."""
        text = "序論\n内容。\n本論\n内容。\n結論\n内容。"
        headings = _detect_japanese_headings(text)
        assert len(headings) >= 3

    def test_numbered_heading_pattern(self) -> None:
        r"""Detect '1. ', '2. ' etc."""
        text = "概要。\n1. はじめに\n内容。\n2. 方法論\n内容。"
        headings = _detect_japanese_headings(text)
        assert len(headings) >= 2

    def test_parenthesized_number_pattern(self) -> None:
        """Detect '（1）', '（2）' etc."""
        text = "概要。\n（1）第一の論点\n内容。\n（2）第二の論点\n内容。"
        headings = _detect_japanese_headings(text)
        assert len(headings) >= 2

    def test_kanji_enumeration_pattern(self) -> None:
        """Detect '一、', '二、' etc."""
        text = "概要。\n一、第一項\n内容。\n二、第二項\n内容。"
        headings = _detect_japanese_headings(text)
        assert len(headings) >= 2

    def test_no_headings_returns_empty(self) -> None:
        """Text without Japanese headings should return empty list."""
        text = "これは普通の文章です。特別な見出しはありません。"
        headings = _detect_japanese_headings(text)
        assert headings == []

    def test_heading_level_assignment(self) -> None:
        """'第X章' should get level 1, numbered items should get level 2."""
        text = "第1章 概要\n内容。\n1. 詳細\n内容。"
        headings = _detect_japanese_headings(text)
        chapter = [h for h in headings if "第1章" in h[1]]
        numbered = [h for h in headings if "詳細" in h[1] or h[1].startswith("1.")]
        if chapter:
            assert chapter[0][2] == 1  # level
        if numbered:
            assert numbered[0][2] == 2  # level


# ============================================================
# _subsplit_section tests
# ============================================================


class TestSubsplitSection:
    """Test sub-splitting of oversized sections."""

    def test_small_section_not_split(self) -> None:
        """Section within max_chars should return as-is."""
        section = Section(
            id="section-0",
            heading="テスト",
            level=1,
            breadcrumb="テスト",
            text="短いテキスト。",
            page_range="",
            char_count=7,
        )
        result = _subsplit_section(section, max_chars=1000)
        assert len(result) == 1
        assert result[0] is section

    def test_oversized_section_splits_at_paragraphs(self) -> None:
        """Oversized section should split at paragraph boundaries."""
        paragraphs = [f"段落{i}。" + "あ" * 30 for i in range(5)]
        text = "# 見出し\n\n" + "\n\n".join(paragraphs)
        section = Section(
            id="section-0",
            heading="見出し",
            level=1,
            breadcrumb="見出し",
            text=text,
            page_range="",
            char_count=len(text),
        )
        result = _subsplit_section(section, max_chars=80)
        assert len(result) > 1

    def test_subsplit_preserves_all_content(self) -> None:
        """All text should be preserved across sub-sections."""
        paragraphs = [f"段落{i}のテキスト。" for i in range(10)]
        text = "\n\n".join(paragraphs)
        section = Section(
            id="section-0",
            heading="テスト",
            level=1,
            breadcrumb="テスト",
            text=text,
            page_range="",
            char_count=len(text),
        )
        result = _subsplit_section(section, max_chars=50)
        combined = "\n\n".join(s.text for s in result)
        for para in paragraphs:
            assert para in combined

    def test_subsplit_ids_sequential(self) -> None:
        """Sub-split sections should have sequential sub-IDs."""
        text = "\n\n".join(["あ" * 30 for _ in range(5)])
        section = Section(
            id="section-2",
            heading="テスト",
            level=1,
            breadcrumb="テスト",
            text=text,
            page_range="",
            char_count=len(text),
        )
        result = _subsplit_section(section, max_chars=40)
        for i, s in enumerate(result):
            assert s.id == f"section-2-{i}"

    def test_subsplit_inherits_metadata(self) -> None:
        """Sub-split sections should inherit level, breadcrumb, page_range."""
        text = "\n\n".join(["あ" * 30 for _ in range(5)])
        section = Section(
            id="section-0",
            heading="テスト",
            level=2,
            breadcrumb="本論 > テスト",
            text=text,
            page_range="pp.5-10",
            char_count=len(text),
        )
        result = _subsplit_section(section, max_chars=40)
        for s in result:
            assert s.level == 2
            assert s.breadcrumb == "本論 > テスト"
            assert s.page_range == "pp.5-10"


# ============================================================
# extract_page_ranges tests
# ============================================================


class TestExtractPageRanges:
    """Test page range extraction from pymupdf4llm page_chunks."""

    def test_single_toc_item(self) -> None:
        """Single TOC item should produce one page range."""
        chunks = [
            {
                "metadata": {"page": 0},
                "toc_items": [{"title": "第1章", "page": 3}],
            },
            {
                "metadata": {"page": 1},
                "toc_items": [{"title": "第1章", "page": 3}],
            },
            {
                "metadata": {"page": 2},
                "toc_items": [{"title": "第2章", "page": 18}],
            },
        ]
        ranges = extract_page_ranges(chunks)
        assert "第1章" in ranges

    def test_empty_chunks(self) -> None:
        """Empty page_chunks should return empty dict."""
        ranges = extract_page_ranges([])
        assert ranges == {}

    def test_no_toc_items(self) -> None:
        """Chunks without toc_items should return empty dict."""
        chunks = [
            {"metadata": {"page": 0}},
            {"metadata": {"page": 1}},
        ]
        ranges = extract_page_ranges(chunks)
        assert ranges == {}

    def test_multiple_toc_items(self) -> None:
        """Multiple TOC items across pages should all be mapped."""
        chunks = [
            {
                "metadata": {"page": 0},
                "toc_items": [{"title": "序論", "page": 1}],
            },
            {
                "metadata": {"page": 5},
                "toc_items": [{"title": "本論", "page": 6}],
            },
            {
                "metadata": {"page": 10},
                "toc_items": [{"title": "結論", "page": 11}],
            },
        ]
        ranges = extract_page_ranges(chunks)
        assert len(ranges) >= 3


# ============================================================
# split_by_headings with Japanese fallback
# ============================================================


class TestSplitByHeadingsJapaneseFallback:
    """Test that split_by_headings falls back to Japanese heading detection."""

    def test_japanese_chapters_without_markdown_headings(self) -> None:
        """Text with Japanese chapter patterns but no # headings should use fallback."""
        text = (
            "前書き。\n\n"
            "第1章 論書名の意味\n\n論書名の内容がここに入る。\n\n"
            "第2章 著者について\n\n著者の経歴がここに入る。"
        )
        sections = split_by_headings(text)
        assert len(sections) >= 2

    def test_japanese_fallback_with_document_title(self) -> None:
        """Japanese fallback should respect document_title in breadcrumbs."""
        text = "第1章 概要\n\n内容。\n\n第2章 詳細\n\n内容。"
        sections = split_by_headings(text, document_title="テスト文書")
        for section in sections:
            if section.heading:
                assert "テスト文書" in section.breadcrumb
