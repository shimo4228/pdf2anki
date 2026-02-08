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
    build_user_prompt,
)

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
