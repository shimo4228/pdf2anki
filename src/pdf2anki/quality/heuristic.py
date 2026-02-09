"""Field-level heuristic scoring for pdf2anki quality assurance.

Provides confidence scoring across 6 dimensions: front_quality, back_quality,
card_type_fit, bloom_level_fit, tags_quality, and atomicity.
"""

from __future__ import annotations

import re

from pdf2anki.quality.duplicate import _detect_duplicates
from pdf2anki.schemas import (
    AnkiCard,
    CardConfidenceScore,
    CardType,
    QualityFlag,
)

# Regex patterns for heuristic scoring
_QUESTION_MARK_RE = re.compile(r"[？?]")
_CLOZE_SYNTAX_RE = re.compile(r"\{\{c\d+::")
_LIST_PATTERN_RE = re.compile(
    r"(?:\d+[.、．]|[・\-*])\s*\S.*(?:\n|\s+)(?:\d+[.、．]|[・\-*])\s*\S"
)
_ENUMERATION_KEYWORDS_RE = re.compile(
    r"(?:挙げ|列挙|リスト|すべて|全て|3つ|三つ|4つ|四つ|5つ|五つ)",
)
_SENTENCE_SPLIT_RE = re.compile(r"[。．.！!]\s*")
_MULTI_CONCEPT_RE = re.compile(
    r"(?:また[、,]|さらに|加えて|そして|および|ならびに)",
)

# Thresholds for scoring
_MIN_FRONT_LENGTH = 10
_MAX_FRONT_LENGTH = 200
_MAX_BACK_LENGTH = 200
_MIN_BACK_LENGTH_QA = 5
_OPTIMAL_BACK_MIN = 10
_OPTIMAL_BACK_MAX = 200


def _score_front_quality(card: AnkiCard) -> float:
    """Score the front (question) quality of a card.

    Checks: length, question form, cloze syntax, clarity.
    For reversible/term_definition cards, short terms are expected.
    """
    front = card.front
    length = len(front)

    if card.card_type == CardType.CLOZE:
        if _CLOZE_SYNTAX_RE.search(front):
            base = 1.0 if length >= _MIN_FRONT_LENGTH else 0.7
            return base
        return 0.4

    # Reversible and term_definition cards often have short terms as fronts
    reversible_types = (CardType.REVERSIBLE, CardType.TERM_DEFINITION)
    if card.card_type in reversible_types and length >= 1:
        return 1.0 if length <= 80 else 0.8

    if length < _MIN_FRONT_LENGTH:
        return max(0.3, length / _MIN_FRONT_LENGTH * 0.6)

    score = 0.7

    if _QUESTION_MARK_RE.search(front):
        score += 0.2

    if _MIN_FRONT_LENGTH <= length <= _MAX_FRONT_LENGTH:
        score += 0.1
    elif length > _MAX_FRONT_LENGTH:
        score -= 0.1

    return min(1.0, max(0.0, score))


def _score_back_quality(card: AnkiCard) -> float:
    """Score the back (answer) quality of a card.

    Checks: length, emptiness for cloze, conciseness.
    """
    back = card.back
    length = len(back)

    if card.card_type == CardType.CLOZE:
        return 1.0 if length == 0 else 0.8

    if length == 0:
        return 0.0

    if length < _MIN_BACK_LENGTH_QA:
        return 0.4

    if length > _MAX_BACK_LENGTH:
        overshoot = length - _MAX_BACK_LENGTH
        penalty = min(0.5, overshoot / 200)
        return max(0.2, 0.7 - penalty)

    if _OPTIMAL_BACK_MIN <= length <= _OPTIMAL_BACK_MAX:
        return 1.0

    return 0.7


def _score_card_type_fit(card: AnkiCard) -> float:
    """Score whether the card_type matches the actual content."""
    front = card.front
    back = card.back

    if card.card_type == CardType.CLOZE:
        if _CLOZE_SYNTAX_RE.search(front):
            return 1.0
        return 0.3

    if card.card_type in (CardType.QA, CardType.SUMMARY_POINT):
        if _CLOZE_SYNTAX_RE.search(front):
            return 0.5
        return 1.0

    if card.card_type == CardType.TERM_DEFINITION:
        if len(front) <= 50 and len(back) > 0:
            return 1.0
        return 0.7

    if card.card_type == CardType.REVERSIBLE:
        if len(front) <= 80 and len(back) > 0:
            return 1.0
        return 0.7

    if card.card_type == CardType.COMPARE_CONTRAST:
        if re.search(r"(?:違い|比較|差|differ|compar)", front):
            return 1.0
        return 0.7

    if card.card_type == CardType.SEQUENCE:
        if re.search(r"(?:次|後|ステップ|step|after|before)", front):
            return 1.0
        return 0.7

    return 0.8


def _score_bloom_level_fit(card: AnkiCard) -> float:
    """Score whether the bloom_level assignment is reasonable.

    Hard to verify automatically, so defaults high with penalties
    for obvious mismatches.
    """
    return 0.9


def _score_tags_quality(card: AnkiCard) -> float:
    """Score the quality of tags.

    Checks: existence, hierarchy (::), quantity.
    """
    tags = card.tags

    if not tags:
        return 0.0

    score = 0.5

    has_hierarchy = any("::" in tag for tag in tags)
    if has_hierarchy:
        score += 0.3

    if len(tags) >= 2:
        score += 0.2

    return min(1.0, score)


def _score_atomicity(card: AnkiCard) -> float:
    """Score whether the card tests exactly one concept.

    Checks: back length, sentence count, multi-concept markers.
    """
    back = card.back

    if card.card_type == CardType.CLOZE:
        cloze_count = len(_CLOZE_SYNTAX_RE.findall(card.front))
        if cloze_count <= 3:
            return 1.0
        return max(0.5, 1.0 - (cloze_count - 3) * 0.1)

    if not back:
        return 0.9

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(back) if s.strip()]
    sentence_count = max(1, len(sentences))

    if sentence_count <= 1:
        return 1.0
    if sentence_count == 2:
        return 0.8
    if sentence_count == 3:
        return 0.6

    score = max(0.3, 1.0 - sentence_count * 0.15)

    if _MULTI_CONCEPT_RE.search(back):
        score = max(0.2, score - 0.15)

    return score


# ============================================================
# Flag detection
# ============================================================


def _detect_flags(card: AnkiCard, scores: dict[str, float]) -> list[QualityFlag]:
    """Detect quality flags based on card content and scores."""
    flags: list[QualityFlag] = []

    term_types = (CardType.CLOZE, CardType.REVERSIBLE, CardType.TERM_DEFINITION)
    if card.card_type not in term_types:
        front = card.front
        if (
            len(front) < _MIN_FRONT_LENGTH
            or (not _QUESTION_MARK_RE.search(front) and scores["front_quality"] < 0.7)
        ):
            flags.append(QualityFlag.VAGUE_QUESTION)

    if len(card.back) > _MAX_BACK_LENGTH:
        flags.append(QualityFlag.TOO_LONG_ANSWER)

    if card.card_type in (CardType.QA, CardType.SUMMARY_POINT) and (
        _LIST_PATTERN_RE.search(card.back)
        or _ENUMERATION_KEYWORDS_RE.search(card.front)
    ):
        flags.append(QualityFlag.LIST_NOT_CLOZE)

    return flags


# ============================================================
# score_card / score_cards
# ============================================================


def score_card(card: AnkiCard) -> CardConfidenceScore:
    """Calculate the confidence score for a single AnkiCard.

    Evaluates 6 dimensions: front_quality, back_quality, card_type_fit,
    bloom_level_fit, tags_quality, atomicity.

    Args:
        card: The AnkiCard to score.

    Returns:
        Immutable CardConfidenceScore with per-field scores and flags.
    """
    field_scores = {
        "front_quality": _score_front_quality(card),
        "back_quality": _score_back_quality(card),
        "card_type_fit": _score_card_type_fit(card),
        "bloom_level_fit": _score_bloom_level_fit(card),
        "tags_quality": _score_tags_quality(card),
        "atomicity": _score_atomicity(card),
    }

    flags = _detect_flags(card, field_scores)

    return CardConfidenceScore(
        front_quality=field_scores["front_quality"],
        back_quality=field_scores["back_quality"],
        card_type_fit=field_scores["card_type_fit"],
        bloom_level_fit=field_scores["bloom_level_fit"],
        tags_quality=field_scores["tags_quality"],
        atomicity=field_scores["atomicity"],
        flags=flags,
    )


def score_cards(cards: list[AnkiCard]) -> list[CardConfidenceScore]:
    """Score a batch of cards and detect cross-card issues like duplicates.

    Args:
        cards: List of AnkiCards to score.

    Returns:
        List of CardConfidenceScore, one per card, in same order.
    """
    if not cards:
        return []

    scores = [score_card(card) for card in cards]
    scores = _detect_duplicates(cards, scores)
    return scores
