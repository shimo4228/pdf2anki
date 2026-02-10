"""Prompt Evaluation Framework for pdf2anki."""

from __future__ import annotations

from pdf2anki.eval.dataset import (
    EvalCase,
    EvalDataset,
    ExpectedCard,
    load_dataset,
)
from pdf2anki.eval.matcher import (
    CaseResult,
    MatchResult,
    match_cards,
)
from pdf2anki.eval.metrics import (
    EvalMetrics,
    calculate_metrics,
)
from pdf2anki.eval.report import (
    print_comparison_report,
    print_eval_report,
    write_eval_json,
)

__all__ = [
    "CaseResult",
    "EvalCase",
    "EvalDataset",
    "EvalMetrics",
    "ExpectedCard",
    "MatchResult",
    "calculate_metrics",
    "load_dataset",
    "match_cards",
    "print_comparison_report",
    "print_eval_report",
    "write_eval_json",
]
