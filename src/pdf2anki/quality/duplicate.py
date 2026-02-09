"""Duplicate detection logic for pdf2anki quality assurance.

Provides character bigram similarity, token-level Jaccard comparison,
and cross-card duplicate concept detection.
"""

from __future__ import annotations

import re

from pdf2anki.schemas import (
    AnkiCard,
    CardConfidenceScore,
    QualityFlag,
)

_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)


def _char_bigrams(text: str) -> set[str]:
    """Extract character bigrams for similarity comparison.

    Bigrams preserve ordering information, reducing false positives
    compared to single-character sets.
    """
    if len(text) < 2:
        return set(text)
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _tokenize(text: str) -> set[str]:
    """Tokenize text for similarity comparison.

    For space-delimited text (English), splits on whitespace/punctuation.
    For CJK text (Japanese/Chinese/Korean), extracts character bigrams
    since word boundaries are not marked by spaces.
    """
    # Whitespace/punctuation split for non-CJK segments
    tokens = re.split(r"[\s\u3000\u3001\u3002\uff1f?\uff01!,.\-:\uff1a]+", text)
    result = {t for t in tokens if len(t) >= 2}

    # CJK character bigrams for Japanese/Chinese/Korean segments
    cjk_chars = _CJK_RE.findall(text)
    if len(cjk_chars) >= 2:
        for i in range(len(cjk_chars) - 1):
            result.add(cjk_chars[i] + cjk_chars[i + 1])

    return result


def _jaccard(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cards_are_similar(a: AnkiCard, b: AnkiCard) -> bool:
    """Check if two cards cover the same concept.

    Uses character bigram Jaccard on front text (reduces false
    positives from single-char sets), plus token, tag, and back overlap.
    """
    front_sim = _jaccard(_char_bigrams(a.front), _char_bigrams(b.front))

    # Same card type + high front similarity
    if front_sim > 0.7 and a.card_type == b.card_type:
        return True

    # Shared tags + moderate front similarity
    if set(a.tags) == set(b.tags) and front_sim > 0.5:
        return True

    # Token-level overlap (catches similar concepts with different phrasing)
    token_sim = _jaccard(_tokenize(a.front), _tokenize(b.front))
    if token_sim > 0.5 and a.card_type == b.card_type:
        return True

    # Back content similarity for same-type cards
    if a.card_type == b.card_type and a.back and b.back:
        back_sim = _jaccard(_char_bigrams(a.back), _char_bigrams(b.back))
        # Strong back similarity with any front overlap
        if back_sim > 0.4 and front_sim > 0.1:
            return True
        # Shared tags + moderate back similarity
        if set(a.tags) == set(b.tags) and back_sim > 0.35:
            return True

    return False


def _detect_duplicates(
    cards: list[AnkiCard],
    scores: list[CardConfidenceScore],
) -> list[CardConfidenceScore]:
    """Check for duplicate concepts across cards and add flags.

    Compares each pair (i, j) only once (j > i) to halve comparisons.
    """
    dup_indices: set[int] = set()

    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _cards_are_similar(cards[i], cards[j]):
                dup_indices.add(i)
                dup_indices.add(j)

    updated: list[CardConfidenceScore] = []
    for i, score in enumerate(scores):
        if i in dup_indices and QualityFlag.DUPLICATE_CONCEPT not in score.flags:
            updated.append(
                score.model_copy(
                    update={"flags": [*score.flags, QualityFlag.DUPLICATE_CONCEPT]}
                )
            )
        else:
            updated.append(score)

    return updated
