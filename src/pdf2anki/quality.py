"""Quality assurance pipeline for pdf2anki.

Provides confidence scoring, LLM critique, and a full quality pipeline
for Anki card generation. Follows the g-kentei-ios pattern:
  [Score cards] -> High confidence -> Pass through
                -> Low confidence  -> LLM critique -> Improve/Split/Remove

All data models are immutable.
"""

from __future__ import annotations

import json
import logging
import re

import anthropic
from pydantic import BaseModel, Field, ValidationError

from pdf2anki.config import AppConfig
from pdf2anki.cost import CostRecord, CostTracker, estimate_cost
from pdf2anki.prompts import CRITIQUE_PROMPT
from pdf2anki.schemas import (
    AnkiCard,
    CardConfidenceScore,
    CardType,
    QualityFlag,
)

logger = logging.getLogger(__name__)

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


# ============================================================
# QualityReport
# ============================================================


class QualityReport(BaseModel, frozen=True):
    """Summary report from the quality pipeline."""

    total_cards: int
    passed_cards: int
    critiqued_cards: int
    removed_cards: int
    improved_cards: int
    split_cards: int
    final_card_count: int

    @property
    def pass_rate(self) -> float:
        if self.total_cards == 0:
            return 0.0
        return self.passed_cards / self.total_cards


# ============================================================
# Field-level scoring functions
# ============================================================


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
    if card.card_type in (CardType.REVERSIBLE, CardType.TERM_DEFINITION):
        if length >= 1:
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

    if card.card_type in (CardType.QA, CardType.SUMMARY_POINT):
        if _LIST_PATTERN_RE.search(card.back) or _ENUMERATION_KEYWORDS_RE.search(
            card.front
        ):
            flags.append(QualityFlag.LIST_NOT_CLOZE)

    return flags


def _detect_duplicates(
    cards: list[AnkiCard],
    scores: list[CardConfidenceScore],
) -> list[CardConfidenceScore]:
    """Check for duplicate concepts across cards and add flags."""
    updated: list[CardConfidenceScore] = []

    for i, (card, score) in enumerate(zip(cards, scores)):
        is_dup = False
        for j, other in enumerate(cards):
            if i == j:
                continue
            if _cards_are_similar(card, other):
                is_dup = True
                break

        if is_dup and QualityFlag.DUPLICATE_CONCEPT not in score.flags:
            updated.append(
                score.model_copy(
                    update={"flags": [*score.flags, QualityFlag.DUPLICATE_CONCEPT]}
                )
            )
        else:
            updated.append(score)

    return updated


def _tokenize(text: str) -> set[str]:
    """Simple tokenization for similarity comparison.

    Splits on whitespace and common punctuation for both
    Japanese and English text.
    """
    tokens = re.split(r"[\s　、。？?！!,.\-:：]+", text)
    return {t for t in tokens if len(t) >= 2}


def _cards_are_similar(a: AnkiCard, b: AnkiCard) -> bool:
    """Check if two cards cover the same concept.

    Uses character-level Jaccard on front text, plus tag overlap.
    """
    # Character-level Jaccard for front text
    chars_a = set(a.front)
    chars_b = set(b.front)
    if not chars_a or not chars_b:
        return False

    char_intersection = chars_a & chars_b
    char_union = chars_a | chars_b
    char_jaccard = len(char_intersection) / len(char_union) if char_union else 0.0

    # Token-level overlap
    tokens_a = _tokenize(a.front)
    tokens_b = _tokenize(b.front)
    token_intersection = tokens_a & tokens_b
    token_union = tokens_a | tokens_b
    token_jaccard = len(token_intersection) / len(token_union) if token_union else 0.0

    # Same card type + high character similarity
    if char_jaccard > 0.7 and a.card_type == b.card_type:
        return True

    # Shared tags + moderate character similarity
    a_tags = set(a.tags)
    b_tags = set(b.tags)
    if a_tags == b_tags and char_jaccard > 0.5:
        return True

    # High token overlap (catches similar concepts with different phrasing)
    if token_jaccard > 0.5 and a.card_type == b.card_type:
        return True

    # Back content similarity for same-type cards
    if a.card_type == b.card_type and a.back and b.back:
        back_chars_a = set(a.back)
        back_chars_b = set(b.back)
        back_union = back_chars_a | back_chars_b
        back_jaccard = (
            len(back_chars_a & back_chars_b) / len(back_union) if back_union else 0.0
        )
        # High back similarity + moderate front overlap = duplicate
        if back_jaccard > 0.55 and char_jaccard > 0.3:
            return True

    return False


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


# ============================================================
# LLM Critique
# ============================================================

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _call_critique_api(
    *,
    client: anthropic.Anthropic,
    model: str,
    cards_json: str,
    source_text: str,
) -> anthropic.types.Message:
    """Call Claude API for card critique.

    Args:
        client: Anthropic client.
        model: Model ID.
        cards_json: JSON string of cards to critique.
        source_text: Original source text for hallucination check.

    Returns:
        Claude API Message response.
    """
    user_content = (
        f"## Cards to Review\n\n{cards_json}\n\n"
        f"## Original Source Text\n\n{source_text[:3000]}"
    )
    return client.messages.create(
        model=model,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": CRITIQUE_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )


def _parse_critique_response(response_text: str) -> list[dict]:
    """Parse and validate the LLM critique response JSON."""
    text = response_text.strip()

    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse critique response: %s", e)
        return []

    if not isinstance(data, list):
        logger.warning("Critique response is not a list")
        return []

    validated: list[dict] = []
    for review in data:
        if not isinstance(review, dict):
            continue
        if "card_index" not in review or "action" not in review:
            logger.warning("Review missing required fields: %s", review)
            continue
        validated.append(review)

    return validated


def critique_cards(
    *,
    cards: list[AnkiCard],
    source_text: str,
    cost_tracker: CostTracker,
    model: str = "claude-sonnet-4-5-20250929",
) -> tuple[list[AnkiCard], CostTracker]:
    """Send low-confidence cards to LLM for critique and improvement.

    Args:
        cards: Cards to critique.
        source_text: Original source text for hallucination checking.
        cost_tracker: Cost tracker for budget enforcement.
        model: Claude model to use for critique.

    Returns:
        Tuple of (improved cards, updated CostTracker).
    """
    if not cards:
        return [], cost_tracker

    cards_data = [card.model_dump() for card in cards]
    cards_json = json.dumps(cards_data, ensure_ascii=False, indent=2)

    try:
        client = anthropic.Anthropic()
        response = _call_critique_api(
            client=client,
            model=model,
            cards_json=cards_json,
            source_text=source_text,
        )
    except (anthropic.APIError, Exception) as e:
        logger.error("API error during critique: %s", e)
        return list(cards), cost_tracker

    cost = estimate_cost(
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    record = CostRecord(
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=cost,
    )
    cost_tracker = cost_tracker.add(record)

    if not response.content:
        logger.warning("Empty critique response")
        return list(cards), cost_tracker

    response_text = response.content[0].text
    reviews = _parse_critique_response(response_text)

    reviewed_indices: set[int] = set()
    result_cards: list[AnkiCard] = []

    for review in reviews:
        idx = review.get("card_index")
        action = review.get("action", "")
        improved = review.get("improved_cards")

        if idx is None or not isinstance(idx, int):
            continue

        reviewed_indices.add(idx)

        if action == "remove":
            continue

        if action in ("improve", "split") and isinstance(improved, list):
            for item in improved:
                try:
                    new_card = AnkiCard.model_validate(item)
                    result_cards.append(new_card)
                except (ValidationError, TypeError) as e:
                    logger.warning("Skipping invalid improved card: %s", e)
        else:
            if 0 <= idx < len(cards):
                result_cards.append(cards[idx])

    for i, card in enumerate(cards):
        if i not in reviewed_indices:
            result_cards.append(card)

    return result_cards, cost_tracker


# ============================================================
# Quality Pipeline
# ============================================================


def run_quality_pipeline(
    *,
    cards: list[AnkiCard],
    source_text: str,
    config: AppConfig,
    cost_tracker: CostTracker,
) -> tuple[list[AnkiCard], QualityReport, CostTracker]:
    """Run the full quality assurance pipeline.

    1. Score all cards
    2. High confidence (>= threshold) -> pass through
    3. Low confidence (< threshold) -> LLM critique (if enabled)
    4. Return final cards + report

    Args:
        cards: Cards to process.
        source_text: Original source text.
        config: Application config with quality settings.
        cost_tracker: Cost tracker for budget.

    Returns:
        Tuple of (final cards, QualityReport, updated CostTracker).
    """
    if not cards:
        report = QualityReport(
            total_cards=0,
            passed_cards=0,
            critiqued_cards=0,
            removed_cards=0,
            improved_cards=0,
            split_cards=0,
            final_card_count=0,
        )
        return [], report, cost_tracker

    threshold = config.quality_confidence_threshold
    scores = score_cards(cards)

    passed: list[AnkiCard] = []
    low_confidence: list[AnkiCard] = []

    for card, score in zip(cards, scores):
        if score.passes_threshold(threshold):
            passed.append(card)
        else:
            low_confidence.append(card)

    passed_count = len(passed)

    improved_count = 0
    removed_count = 0
    split_count = 0
    critiqued_count = 0
    critique_result_cards: list[AnkiCard] = []

    enable_critique = config.quality_enable_critique
    max_rounds = config.quality_max_critique_rounds

    if enable_critique and low_confidence and max_rounds > 0:
        critique_result_cards, cost_tracker = critique_cards(
            cards=low_confidence,
            source_text=source_text,
            cost_tracker=cost_tracker,
            model=config.model,
        )

        critiqued_count = len(low_confidence)
        original_count = len(low_confidence)
        result_count = len(critique_result_cards)

        if result_count < original_count:
            removed_count = original_count - result_count
        if result_count > original_count:
            split_count = result_count - original_count
        improved_count = min(result_count, original_count)
    else:
        critique_result_cards = low_confidence

    final_cards = [*passed, *critique_result_cards]

    report = QualityReport(
        total_cards=len(cards),
        passed_cards=passed_count,
        critiqued_cards=critiqued_count,
        removed_cards=removed_count,
        improved_cards=improved_count,
        split_cards=split_count,
        final_card_count=len(final_cards),
    )

    return final_cards, report, cost_tracker
