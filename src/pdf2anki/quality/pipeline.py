"""Quality pipeline orchestration for pdf2anki.

Provides QualityReport and the full quality pipeline that coordinates
heuristic scoring, duplicate detection, and LLM critique.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from pdf2anki.config import AppConfig
from pdf2anki.cost import CostTracker
from pdf2anki.quality.critique import critique_cards
from pdf2anki.quality.heuristic import score_cards
from pdf2anki.schemas import AnkiCard

logger = logging.getLogger(__name__)


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

    for card, score in zip(cards, scores, strict=True):
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
