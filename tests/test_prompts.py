"""Tests for pdf2anki.prompts - TDD RED phase.

Tests cover:
- SYSTEM_PROMPT constant (Wozniak principles, card types, Bloom's taxonomy)
- CRITIQUE_PROMPT constant (evaluation criteria)
- build_user_prompt() function (text input, settings, edge cases)
"""

from __future__ import annotations

import pytest

from pdf2anki.prompts import (
    CRITIQUE_PROMPT,
    SYSTEM_PROMPT,
    build_section_prompt,
    build_user_prompt,
)
from pdf2anki.section import Section

# ============================================================
# SYSTEM_PROMPT Tests
# ============================================================


class TestSystemPrompt:
    """Test SYSTEM_PROMPT constant for Wozniak principles."""

    def test_system_prompt_is_nonempty_string(self) -> None:
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100

    def test_contains_wozniak_minimum_information_principle(self) -> None:
        """Wozniak principle: minimum information per card."""
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "minimum information" in prompt_lower or "最小情報" in prompt_lower

    def test_contains_cloze_conversion_rule(self) -> None:
        """Wozniak principle: lists should become cloze deletions."""
        assert "cloze" in SYSTEM_PROMPT.lower()

    def test_contains_redundancy_principle(self) -> None:
        """Wozniak principle: use redundancy (same concept from multiple angles)."""
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "redundancy" in prompt_lower or "冗長" in prompt_lower

    def test_contains_mnemonic_hint(self) -> None:
        """Wozniak principle: mnemonic hints."""
        assert "mnemonic" in SYSTEM_PROMPT.lower()

    def test_contains_all_eight_card_types(self) -> None:
        """SYSTEM_PROMPT should reference all 8 card types."""
        card_types = [
            "qa",
            "term_definition",
            "summary_point",
            "cloze",
            "reversible",
            "sequence",
            "compare_contrast",
            "image_occlusion",
        ]
        for ct in card_types:
            assert ct in SYSTEM_PROMPT, f"Missing card type: {ct}"

    def test_contains_bloom_levels(self) -> None:
        """SYSTEM_PROMPT should reference Bloom's Taxonomy levels."""
        bloom_levels = [
            "remember",
            "understand",
            "apply",
            "analyze",
            "evaluate",
            "create",
        ]
        for bl in bloom_levels:
            assert bl in SYSTEM_PROMPT.lower(), f"Missing Bloom level: {bl}"

    def test_contains_good_bad_examples(self) -> None:
        """Wozniak principle: show good and bad card examples."""
        prompt_lower = SYSTEM_PROMPT.lower()
        has_examples = (
            ("good" in prompt_lower and "bad" in prompt_lower)
            or ("良い" in SYSTEM_PROMPT and "悪い" in SYSTEM_PROMPT)
        )
        assert has_examples

    def test_contains_json_output_instruction(self) -> None:
        """Should instruct the LLM to output structured JSON."""
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "json" in prompt_lower or "structured" in prompt_lower


# ============================================================
# CRITIQUE_PROMPT Tests
# ============================================================


class TestCritiquePrompt:
    """Test CRITIQUE_PROMPT constant for card evaluation."""

    def test_critique_prompt_is_nonempty_string(self) -> None:
        assert isinstance(CRITIQUE_PROMPT, str)
        assert len(CRITIQUE_PROMPT) > 50

    def test_contains_evaluation_criteria(self) -> None:
        """Should mention quality evaluation criteria."""
        prompt_lower = CRITIQUE_PROMPT.lower()
        assert any(
            keyword in prompt_lower
            for keyword in ["quality", "evaluate", "review", "品質", "評価"]
        )

    def test_contains_improvement_instruction(self) -> None:
        """Should instruct LLM to improve, split, or remove cards."""
        prompt_lower = CRITIQUE_PROMPT.lower()
        assert any(
            keyword in prompt_lower
            for keyword in ["improve", "split", "remove", "改善", "分割", "除去"]
        )

    def test_contains_minimum_information_check(self) -> None:
        """Should check for atomic (1 concept per card) principle."""
        prompt_lower = CRITIQUE_PROMPT.lower()
        assert any(
            keyword in prompt_lower
            for keyword in ["atomic", "one concept", "1概念", "原子"]
        )

    def test_contains_flag_reference(self) -> None:
        """Should reference quality flags."""
        prompt_lower = CRITIQUE_PROMPT.lower()
        assert any(
            keyword in prompt_lower
            for keyword in ["flag", "vague", "too_long", "フラグ"]
        )


# ============================================================
# build_user_prompt Tests
# ============================================================


class TestBuildUserPrompt:
    """Test build_user_prompt function."""

    def test_basic_prompt_contains_text(self) -> None:
        """User prompt should contain the source text."""
        text = "ニューラルネットワークの基礎概念について説明します。"
        prompt = build_user_prompt(text)
        assert text in prompt

    def test_basic_prompt_is_string(self) -> None:
        prompt = build_user_prompt("some text")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_max_cards_included(self) -> None:
        """Should include max_cards setting in prompt."""
        prompt = build_user_prompt("text", max_cards=30)
        assert "30" in prompt

    def test_default_max_cards(self) -> None:
        """Default max_cards should appear in prompt."""
        prompt = build_user_prompt("text")
        assert "50" in prompt

    def test_card_types_filter_included(self) -> None:
        """When specific card types are requested, they should appear."""
        prompt = build_user_prompt("text", card_types=["qa", "cloze"])
        assert "qa" in prompt
        assert "cloze" in prompt

    def test_focus_topics_included(self) -> None:
        """Focus topics should be included in prompt."""
        prompt = build_user_prompt("text", focus_topics=["CNN", "RNN"])
        assert "CNN" in prompt
        assert "RNN" in prompt

    def test_bloom_filter_included(self) -> None:
        """Bloom level filter should be included in prompt."""
        prompt = build_user_prompt("text", bloom_filter=["remember", "understand"])
        assert "remember" in prompt
        assert "understand" in prompt

    def test_additional_tags_included(self) -> None:
        """Additional tags should be included in prompt."""
        prompt = build_user_prompt("text", additional_tags=["AI::基礎", "ML"])
        assert "AI::基礎" in prompt
        assert "ML" in prompt

    def test_empty_text_raises(self) -> None:
        """Empty text should raise ValueError."""
        with pytest.raises(ValueError, match="text"):
            build_user_prompt("")

    def test_whitespace_only_text_raises(self) -> None:
        """Whitespace-only text should raise ValueError."""
        with pytest.raises(ValueError, match="text"):
            build_user_prompt("   \n\t  ")

    def test_no_optional_params(self) -> None:
        """Should work without any optional parameters."""
        prompt = build_user_prompt("Source text here.")
        assert "Source text here." in prompt

    def test_empty_focus_topics_ignored(self) -> None:
        """Empty focus topics list should not add focus section."""
        prompt_without = build_user_prompt("text")
        prompt_with_empty = build_user_prompt("text", focus_topics=[])
        # Both should be equivalent (no focus section)
        assert len(prompt_without) == len(prompt_with_empty)

    def test_long_text_included_fully(self) -> None:
        """Long text should be included without truncation."""
        long_text = "A" * 10000
        prompt = build_user_prompt(long_text)
        assert long_text in prompt


# ============================================================
# build_section_prompt Tests
# ============================================================


def _make_section(
    *,
    heading: str = "第1章 論書名の意味",
    level: int = 1,
    breadcrumb: str = "正理の海 > 本論 > 第1章",
    text: str = "# 第1章 論書名の意味\n\n論書名の内容がここに入る。",
    page_range: str = "pp.3-18",
) -> Section:
    """Helper to create a Section for prompt tests."""
    return Section(
        id="section-1",
        heading=heading,
        level=level,
        breadcrumb=breadcrumb,
        text=text,
        page_range=page_range,
        char_count=len(text),
    )


class TestBuildSectionPrompt:
    """Test build_section_prompt function for section-aware prompting."""

    def test_contains_section_text(self) -> None:
        """Prompt should contain the section body text."""
        section = _make_section()
        prompt = build_section_prompt(section)
        assert "論書名の内容がここに入る" in prompt

    def test_contains_breadcrumb(self) -> None:
        """Prompt should include the breadcrumb context."""
        section = _make_section(breadcrumb="正理の海 > 本論 > 第1章")
        prompt = build_section_prompt(section)
        assert "正理の海 > 本論 > 第1章" in prompt

    def test_contains_document_title(self) -> None:
        """Prompt should include document_title when provided."""
        section = _make_section()
        prompt = build_section_prompt(section, document_title="正理の海全編")
        assert "正理の海全編" in prompt

    def test_contains_page_range(self) -> None:
        """Prompt should include page range when available."""
        section = _make_section(page_range="pp.3-18")
        prompt = build_section_prompt(section)
        assert "pp.3-18" in prompt

    def test_empty_page_range_omitted(self) -> None:
        """Empty page_range should not add a page range line."""
        section = _make_section(page_range="")
        prompt = build_section_prompt(section)
        # Should not contain "Page range" or "pp." placeholder
        assert "pp." not in prompt

    def test_hierarchical_tag_instruction(self) -> None:
        """Prompt should instruct LLM to generate hierarchical tags from breadcrumb."""
        section = _make_section(breadcrumb="正理の海 > 本論 > 第1章")
        prompt = build_section_prompt(section)
        # Should contain tag hierarchy instruction using :: separator
        assert "::" in prompt

    def test_default_max_cards_is_20(self) -> None:
        """Default max_cards should be 20 (section-level, not document-level 50)."""
        section = _make_section()
        prompt = build_section_prompt(section)
        assert "20" in prompt

    def test_custom_max_cards(self) -> None:
        """Custom max_cards should override default."""
        section = _make_section()
        prompt = build_section_prompt(section, max_cards=10)
        assert "10" in prompt

    def test_card_types_included(self) -> None:
        """Card type filter should appear in prompt."""
        section = _make_section()
        prompt = build_section_prompt(section, card_types=["qa", "cloze"])
        assert "qa" in prompt
        assert "cloze" in prompt

    def test_focus_topics_included(self) -> None:
        """Focus topics should appear in prompt."""
        section = _make_section()
        prompt = build_section_prompt(section, focus_topics=["因明", "論理学"])
        assert "因明" in prompt
        assert "論理学" in prompt

    def test_bloom_filter_included(self) -> None:
        """Bloom level filter should appear in prompt."""
        section = _make_section()
        prompt = build_section_prompt(
            section, bloom_filter=["remember", "understand"]
        )
        assert "remember" in prompt
        assert "understand" in prompt

    def test_additional_tags_included(self) -> None:
        """Additional tags should appear in prompt."""
        section = _make_section()
        prompt = build_section_prompt(
            section, additional_tags=["仏教::インド", "哲学"]
        )
        assert "仏教::インド" in prompt
        assert "哲学" in prompt

    def test_empty_section_text_raises(self) -> None:
        """Section with empty text should raise ValueError."""
        section = Section(
            id="section-0",
            heading="",
            level=0,
            breadcrumb="",
            text="",
            page_range="",
            char_count=0,
        )
        with pytest.raises(ValueError, match="text"):
            build_section_prompt(section)

    def test_returns_string(self) -> None:
        """Should return a non-empty string."""
        section = _make_section()
        prompt = build_section_prompt(section)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_preamble_section_no_breadcrumb(self) -> None:
        """Level-0 preamble section with empty breadcrumb should work."""
        section = Section(
            id="section-0",
            heading="",
            level=0,
            breadcrumb="",
            text="前書きのテキスト。",
            page_range="",
            char_count=9,
        )
        prompt = build_section_prompt(section)
        assert "前書きのテキスト" in prompt
