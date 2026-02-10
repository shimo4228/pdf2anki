"""Evaluation metrics calculation.

Computes Recall, Precision, F1, and aggregate statistics
from a list of CaseResult objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from pdf2anki.eval.matcher import CaseResult


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Aggregate evaluation metrics."""

    recall: float
    precision: float
    f1: float
    avg_similarity: float
    total_cost_usd: float
    total_expected: int
    total_generated: int
    total_matched: int


def calculate_metrics(
    case_results: list[CaseResult],
    *,
    cost_usd: float = 0.0,
) -> EvalMetrics:
    """Calculate aggregate metrics from case results.

    Args:
        case_results: List of CaseResult from matching.
        cost_usd: Total API cost for the evaluation run.

    Returns:
        EvalMetrics with recall, precision, F1, etc.
    """
    total_expected = 0
    total_generated = 0
    total_matched = 0
    similarity_sum = 0.0

    for cr in case_results:
        total_expected += len(cr.matches)
        total_generated += len(cr.generated_cards)
        for m in cr.matches:
            if m.matched_card is not None:
                total_matched += 1
                similarity_sum += m.similarity

    recall = (
        total_matched / total_expected if total_expected > 0 else 0.0
    )
    precision = (
        total_matched / total_generated if total_generated > 0 else 0.0
    )
    f1 = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0
        else 0.0
    )
    avg_sim = (
        similarity_sum / total_matched if total_matched > 0 else 0.0
    )

    return EvalMetrics(
        recall=recall,
        precision=precision,
        f1=f1,
        avg_similarity=avg_sim,
        total_cost_usd=cost_usd,
        total_expected=total_expected,
        total_generated=total_generated,
        total_matched=total_matched,
    )
