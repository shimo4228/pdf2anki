"""Card matching logic for evaluation.

Matches expected cards (keyword-based) against generated AnkiCards
using keyword overlap similarity with optional card-type bonus.
"""

from __future__ import annotations

from dataclasses import dataclass

from pdf2anki.eval.dataset import ExpectedCard
from pdf2anki.schemas import AnkiCard


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Result of matching a single expected card."""

    expected: ExpectedCard
    matched_card: AnkiCard | None
    similarity: float


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Result of evaluating a single case."""

    case_id: str
    generated_cards: tuple[AnkiCard, ...]
    matches: tuple[MatchResult, ...]
    unmatched_generated: tuple[AnkiCard, ...]


def _keyword_similarity(keywords: list[str], text: str) -> float:
    """Calculate fraction of keywords found in text.

    Returns 0.0 if keywords list is empty.
    """
    if not keywords:
        return 0.0
    found = sum(1 for kw in keywords if kw in text)
    return found / len(keywords)


def _score_pair(expected: ExpectedCard, card: AnkiCard) -> float:
    """Score similarity between an expected card and a generated card.

    Scoring:
    - 40% front keyword overlap
    - 40% back keyword overlap
    - 20% card type match bonus (if expected type is specified)
    """
    front_sim = _keyword_similarity(expected.front_keywords, card.front)
    back_sim = _keyword_similarity(expected.back_keywords, card.back)

    type_bonus = 0.0
    if expected.card_type is not None:
        type_bonus = 1.0 if card.card_type == expected.card_type else 0.0

    if expected.card_type is not None:
        return front_sim * 0.4 + back_sim * 0.4 + type_bonus * 0.2
    return front_sim * 0.5 + back_sim * 0.5


def match_cards(
    expected: list[ExpectedCard],
    generated: list[AnkiCard],
    *,
    case_id: str,
    threshold: float = 0.3,
) -> CaseResult:
    """Match expected cards against generated cards.

    Uses greedy best-match: for each expected card, find the
    highest-scoring generated card above threshold.
    Each generated card can match at most one expected card.

    Args:
        expected: Expected card definitions.
        generated: Generated AnkiCards to evaluate.
        case_id: Identifier for the case.
        threshold: Minimum similarity to count as a match.

    Returns:
        CaseResult with match details.
    """
    used_indices: set[int] = set()
    matches: list[MatchResult] = []

    for ec in expected:
        best_score = 0.0
        best_idx = -1

        for i, card in enumerate(generated):
            if i in used_indices:
                continue
            score = _score_pair(ec, card)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0 and best_score >= threshold:
            used_indices.add(best_idx)
            matches.append(
                MatchResult(
                    expected=ec,
                    matched_card=generated[best_idx],
                    similarity=best_score,
                )
            )
        else:
            matches.append(
                MatchResult(expected=ec, matched_card=None, similarity=0.0)
            )

    unmatched = tuple(
        card for i, card in enumerate(generated) if i not in used_indices
    )

    return CaseResult(
        case_id=case_id,
        generated_cards=tuple(generated),
        matches=tuple(matches),
        unmatched_generated=unmatched,
    )
