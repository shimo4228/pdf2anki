"""Pydantic schemas for pdf2anki card generation.

Defines the core data models: CardType, BloomLevel, AnkiCard,
CardConfidenceScore, QualityFlag, and ExtractionResult.
All models use frozen=True for immutability.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CardType(StrEnum):
    """8 card types supported by pdf2anki."""

    QA = "qa"
    TERM_DEFINITION = "term_definition"
    SUMMARY_POINT = "summary_point"
    CLOZE = "cloze"
    REVERSIBLE = "reversible"
    SEQUENCE = "sequence"
    COMPARE_CONTRAST = "compare_contrast"
    IMAGE_OCCLUSION = "image_occlusion"


class BloomLevel(StrEnum):
    """Bloom's Taxonomy cognitive levels (ordered low to high)."""

    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class QualityFlag(StrEnum):
    """Quality issue flags detected during confidence scoring."""

    VAGUE_QUESTION = "vague_question"
    TOO_LONG_ANSWER = "too_long_answer"
    LIST_NOT_CLOZE = "list_not_cloze"
    DUPLICATE_CONCEPT = "duplicate_concept"
    TOO_SIMPLE = "too_simple"
    HALLUCINATION_RISK = "hallucination_risk"


class AnkiCard(BaseModel, frozen=True):
    """A single Anki flashcard. Immutable."""

    front: str = Field(min_length=1)
    back: str = Field(default="")
    card_type: CardType
    bloom_level: BloomLevel
    tags: list[str] = Field(min_length=1)
    related_concepts: list[str] = Field(default_factory=list)
    mnemonic_hint: str | None = Field(default=None)
    media: list[str] = Field(default_factory=list)


# Weights for confidence scoring (from plan)
_CONFIDENCE_WEIGHTS = {
    "front_quality": 0.25,
    "back_quality": 0.25,
    "card_type_fit": 0.15,
    "bloom_level_fit": 0.10,
    "tags_quality": 0.10,
    "atomicity": 0.15,
}


class CardConfidenceScore(BaseModel, frozen=True):
    """Confidence score breakdown for a card."""

    front_quality: float = Field(ge=0.0, le=1.0)
    back_quality: float = Field(ge=0.0, le=1.0)
    card_type_fit: float = Field(ge=0.0, le=1.0)
    bloom_level_fit: float = Field(ge=0.0, le=1.0)
    tags_quality: float = Field(ge=0.0, le=1.0)
    atomicity: float = Field(ge=0.0, le=1.0)
    flags: list[QualityFlag] = Field(default_factory=list)

    @property
    def weighted_total(self) -> float:
        """Calculate the weighted total confidence score."""
        return (
            self.front_quality * _CONFIDENCE_WEIGHTS["front_quality"]
            + self.back_quality * _CONFIDENCE_WEIGHTS["back_quality"]
            + self.card_type_fit * _CONFIDENCE_WEIGHTS["card_type_fit"]
            + self.bloom_level_fit * _CONFIDENCE_WEIGHTS["bloom_level_fit"]
            + self.tags_quality * _CONFIDENCE_WEIGHTS["tags_quality"]
            + self.atomicity * _CONFIDENCE_WEIGHTS["atomicity"]
        )

    def passes_threshold(self, threshold: float) -> bool:
        """Check if the weighted total meets the given threshold."""
        return self.weighted_total >= threshold


class ExtractionResult(BaseModel, frozen=True):
    """Result of extracting cards from a document."""

    source_file: str
    cards: list[AnkiCard] = Field(default_factory=list)
    model_used: str

    @property
    def card_count(self) -> int:
        """Return the number of extracted cards."""
        return len(self.cards)

    def cards_by_type(self, card_type: CardType) -> list[AnkiCard]:
        """Filter cards by card type."""
        return [c for c in self.cards if c.card_type == card_type]

    def cards_by_bloom(self, bloom_level: BloomLevel) -> list[AnkiCard]:
        """Filter cards by Bloom's Taxonomy level."""
        return [c for c in self.cards if c.bloom_level == bloom_level]
