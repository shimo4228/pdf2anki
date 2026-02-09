"""Tests for pdf2anki.schemas - TDD RED phase.

Tests cover:
- CardType enum (8 types)
- BloomLevel enum (6 levels)
- QualityFlag enum (6 flags)
- AnkiCard model (immutability, validation, defaults)
- CardConfidenceScore model (scoring, weighted total)
- ExtractionResult model (card collection, metadata)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pdf2anki.schemas import (
    AnkiCard,
    BloomLevel,
    CardConfidenceScore,
    CardType,
    ExtractionResult,
    QualityFlag,
)

# ============================================================
# CardType Enum Tests
# ============================================================


class TestCardType:
    """Test the 8 card types."""

    def test_all_card_types_exist(self) -> None:
        expected = {
            "qa",
            "term_definition",
            "summary_point",
            "cloze",
            "reversible",
            "sequence",
            "compare_contrast",
            "image_occlusion",
        }
        actual = {ct.value for ct in CardType}
        assert actual == expected

    def test_card_type_count(self) -> None:
        assert len(CardType) == 8

    def test_card_type_is_string(self) -> None:
        assert CardType.QA == "qa"
        assert CardType.CLOZE == "cloze"


# ============================================================
# BloomLevel Enum Tests
# ============================================================


class TestBloomLevel:
    """Test Bloom's Taxonomy levels."""

    def test_all_bloom_levels_exist(self) -> None:
        expected = {
            "remember",
            "understand",
            "apply",
            "analyze",
            "evaluate",
            "create",
        }
        actual = {bl.value for bl in BloomLevel}
        assert actual == expected

    def test_bloom_level_count(self) -> None:
        assert len(BloomLevel) == 6

    def test_bloom_level_ordering(self) -> None:
        """Bloom levels should have a natural ordering via their index."""
        levels = list(BloomLevel)
        assert levels[0] == BloomLevel.REMEMBER
        assert levels[-1] == BloomLevel.CREATE


# ============================================================
# QualityFlag Enum Tests
# ============================================================


class TestQualityFlag:
    """Test quality issue flags."""

    def test_all_flags_exist(self) -> None:
        expected = {
            "vague_question",
            "too_long_answer",
            "list_not_cloze",
            "duplicate_concept",
            "too_simple",
            "hallucination_risk",
        }
        actual = {qf.value for qf in QualityFlag}
        assert actual == expected

    def test_flag_count(self) -> None:
        assert len(QualityFlag) == 6


# ============================================================
# AnkiCard Model Tests
# ============================================================


class TestAnkiCard:
    """Test AnkiCard Pydantic model."""

    def test_create_qa_card(self, sample_qa_card: AnkiCard) -> None:
        expected = "ニューラルネットワークの活性化関数の役割は何ですか？"
        assert sample_qa_card.front == expected
        assert sample_qa_card.card_type == CardType.QA
        assert sample_qa_card.bloom_level == BloomLevel.UNDERSTAND
        assert "AI::基礎" in sample_qa_card.tags

    def test_create_cloze_card(self, sample_cloze_card: AnkiCard) -> None:
        assert "{{c1::" in sample_cloze_card.front
        assert sample_cloze_card.card_type == CardType.CLOZE
        assert sample_cloze_card.back == ""

    def test_create_reversible_card(self, sample_reversible_card: AnkiCard) -> None:
        assert sample_reversible_card.card_type == CardType.REVERSIBLE
        assert "sigmoid" in sample_reversible_card.related_concepts
        assert sample_reversible_card.mnemonic_hint is not None

    def test_default_optional_fields(self) -> None:
        card = AnkiCard(
            front="What is X?",
            back="X is Y.",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        assert card.related_concepts == []
        assert card.mnemonic_hint is None

    def test_frozen_immutability(self, sample_qa_card: AnkiCard) -> None:
        """AnkiCard should be frozen (immutable)."""
        with pytest.raises(ValidationError):
            sample_qa_card.front = "modified"  # type: ignore[misc]

    def test_empty_front_rejected(self) -> None:
        """Front text cannot be empty."""
        with pytest.raises(ValidationError):
            AnkiCard(
                front="",
                back="answer",
                card_type=CardType.QA,
                bloom_level=BloomLevel.REMEMBER,
                tags=["test"],
            )

    def test_tags_must_have_at_least_one(self) -> None:
        """Tags list must contain at least one tag."""
        with pytest.raises(ValidationError):
            AnkiCard(
                front="question",
                back="answer",
                card_type=CardType.QA,
                bloom_level=BloomLevel.REMEMBER,
                tags=[],
            )

    def test_serialization_roundtrip(self, sample_qa_card: AnkiCard) -> None:
        """Card should survive JSON serialization roundtrip."""
        json_str = sample_qa_card.model_dump_json()
        restored = AnkiCard.model_validate_json(json_str)
        assert restored == sample_qa_card


# ============================================================
# CardConfidenceScore Model Tests
# ============================================================


class TestCardConfidenceScore:
    """Test confidence scoring model."""

    def test_create_high_confidence_score(self) -> None:
        score = CardConfidenceScore(
            front_quality=0.95,
            back_quality=0.90,
            card_type_fit=0.85,
            bloom_level_fit=0.90,
            tags_quality=0.80,
            atomicity=0.95,
            flags=[],
        )
        assert score.front_quality == 0.95
        assert score.flags == []

    def test_weighted_total_high_confidence(self) -> None:
        """Weighted total should reflect the plan's weights:
        front=0.25, back=0.25, type=0.15, bloom=0.10, tags=0.10, atom=0.15
        """
        score = CardConfidenceScore(
            front_quality=1.0,
            back_quality=1.0,
            card_type_fit=1.0,
            bloom_level_fit=1.0,
            tags_quality=1.0,
            atomicity=1.0,
            flags=[],
        )
        assert score.weighted_total == pytest.approx(1.0)

    def test_weighted_total_mixed(self) -> None:
        score = CardConfidenceScore(
            front_quality=0.8,
            back_quality=0.6,
            card_type_fit=0.9,
            bloom_level_fit=0.7,
            tags_quality=0.5,
            atomicity=0.8,
            flags=[QualityFlag.VAGUE_QUESTION],
        )
        # 0.8*0.25 + 0.6*0.25 + 0.9*0.15 + 0.7*0.10 + 0.5*0.10 + 0.8*0.15
        expected = 0.20 + 0.15 + 0.135 + 0.07 + 0.05 + 0.12
        assert score.weighted_total == pytest.approx(expected)

    def test_passes_threshold_high(self) -> None:
        score = CardConfidenceScore(
            front_quality=0.95,
            back_quality=0.95,
            card_type_fit=0.90,
            bloom_level_fit=0.90,
            tags_quality=0.85,
            atomicity=0.90,
            flags=[],
        )
        assert score.passes_threshold(0.90) is True

    def test_fails_threshold_low(self) -> None:
        score = CardConfidenceScore(
            front_quality=0.5,
            back_quality=0.5,
            card_type_fit=0.5,
            bloom_level_fit=0.5,
            tags_quality=0.5,
            atomicity=0.5,
            flags=[QualityFlag.VAGUE_QUESTION],
        )
        assert score.passes_threshold(0.90) is False

    def test_score_fields_clamped_0_to_1(self) -> None:
        """Score fields must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            CardConfidenceScore(
                front_quality=1.5,  # out of range
                back_quality=0.5,
                card_type_fit=0.5,
                bloom_level_fit=0.5,
                tags_quality=0.5,
                atomicity=0.5,
                flags=[],
            )

    def test_score_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CardConfidenceScore(
                front_quality=-0.1,
                back_quality=0.5,
                card_type_fit=0.5,
                bloom_level_fit=0.5,
                tags_quality=0.5,
                atomicity=0.5,
                flags=[],
            )

    def test_frozen_immutability(self) -> None:
        score = CardConfidenceScore(
            front_quality=0.9,
            back_quality=0.9,
            card_type_fit=0.9,
            bloom_level_fit=0.9,
            tags_quality=0.9,
            atomicity=0.9,
            flags=[],
        )
        with pytest.raises(ValidationError):
            score.front_quality = 0.1  # type: ignore[misc]


# ============================================================
# ExtractionResult Model Tests
# ============================================================


class TestExtractionResult:
    """Test extraction result container."""

    def test_create_result(self, sample_qa_card: AnkiCard) -> None:
        result = ExtractionResult(
            source_file="test.pdf",
            cards=[sample_qa_card],
            model_used="claude-sonnet-4-5-20250929",
        )
        assert result.source_file == "test.pdf"
        assert len(result.cards) == 1
        assert result.cards[0].card_type == CardType.QA

    def test_card_count_property(
        self, sample_qa_card: AnkiCard, sample_cloze_card: AnkiCard
    ) -> None:
        result = ExtractionResult(
            source_file="test.pdf",
            cards=[sample_qa_card, sample_cloze_card],
            model_used="claude-sonnet-4-5-20250929",
        )
        assert result.card_count == 2

    def test_empty_cards_allowed(self) -> None:
        result = ExtractionResult(
            source_file="empty.pdf",
            cards=[],
            model_used="claude-sonnet-4-5-20250929",
        )
        assert result.card_count == 0

    def test_cards_by_type(
        self, sample_qa_card: AnkiCard, sample_cloze_card: AnkiCard
    ) -> None:
        result = ExtractionResult(
            source_file="test.pdf",
            cards=[sample_qa_card, sample_cloze_card],
            model_used="claude-sonnet-4-5-20250929",
        )
        qa_cards = result.cards_by_type(CardType.QA)
        assert len(qa_cards) == 1
        assert qa_cards[0].card_type == CardType.QA

        cloze_cards = result.cards_by_type(CardType.CLOZE)
        assert len(cloze_cards) == 1

    def test_cards_by_bloom(self, sample_qa_card: AnkiCard) -> None:
        result = ExtractionResult(
            source_file="test.pdf",
            cards=[sample_qa_card],
            model_used="claude-sonnet-4-5-20250929",
        )
        understand_cards = result.cards_by_bloom(BloomLevel.UNDERSTAND)
        assert len(understand_cards) == 1

        apply_cards = result.cards_by_bloom(BloomLevel.APPLY)
        assert len(apply_cards) == 0

    def test_json_serialization(self, sample_qa_card: AnkiCard) -> None:
        result = ExtractionResult(
            source_file="test.pdf",
            cards=[sample_qa_card],
            model_used="claude-sonnet-4-5-20250929",
        )
        json_str = result.model_dump_json(indent=2)
        restored = ExtractionResult.model_validate_json(json_str)
        assert restored == result

    def test_frozen_immutability(self, sample_qa_card: AnkiCard) -> None:
        result = ExtractionResult(
            source_file="test.pdf",
            cards=[sample_qa_card],
            model_used="claude-sonnet-4-5-20250929",
        )
        with pytest.raises(ValidationError):
            result.source_file = "other.pdf"  # type: ignore[misc]
