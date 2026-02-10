"""Evaluation report generation.

Produces Rich table output and JSON reports for evaluation results.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from pdf2anki.eval.matcher import CaseResult
from pdf2anki.eval.metrics import EvalMetrics


def print_eval_report(
    metrics: EvalMetrics,
    case_results: list[CaseResult],
) -> None:
    """Print evaluation metrics as a Rich table."""
    console = Console()

    table = Table(title="Evaluation Report", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Recall", f"{metrics.recall:.1%}")
    table.add_row("Precision", f"{metrics.precision:.1%}")
    table.add_row("F1 Score", f"{metrics.f1:.1%}")
    table.add_row("Avg Similarity", f"{metrics.avg_similarity:.2f}")
    table.add_row("Total Expected", str(metrics.total_expected))
    table.add_row("Total Generated", str(metrics.total_generated))
    table.add_row("Total Matched", str(metrics.total_matched))
    table.add_row("Cost (USD)", f"${metrics.total_cost_usd:.4f}")

    console.print(table)

    # Per-case breakdown
    if case_results:
        detail = Table(title="Per-Case Results", show_header=True)
        detail.add_column("Case ID", style="cyan")
        detail.add_column("Expected", justify="right")
        detail.add_column("Generated", justify="right")
        detail.add_column("Matched", justify="right")
        detail.add_column("Recall", justify="right", style="green")

        for cr in case_results:
            matched = sum(
                1 for m in cr.matches if m.matched_card is not None
            )
            n_expected = len(cr.matches)
            case_recall = (
                matched / n_expected if n_expected > 0 else 0.0
            )
            detail.add_row(
                cr.case_id,
                str(n_expected),
                str(len(cr.generated_cards)),
                str(matched),
                f"{case_recall:.0%}",
            )

        console.print(detail)


def print_comparison_report(
    metrics_a: EvalMetrics,
    metrics_b: EvalMetrics,
    *,
    label_a: str = "v1",
    label_b: str = "v2",
) -> None:
    """Print side-by-side comparison of two evaluation runs."""
    console = Console()

    table = Table(
        title=f"Comparison: {label_a} vs {label_b}",
        show_header=True,
    )
    table.add_column("Metric", style="cyan")
    table.add_column(label_a, justify="right")
    table.add_column(label_b, justify="right")
    table.add_column("Delta", justify="right", style="yellow")

    rows = [
        ("Recall", metrics_a.recall, metrics_b.recall, True),
        ("Precision", metrics_a.precision, metrics_b.precision, True),
        ("F1 Score", metrics_a.f1, metrics_b.f1, True),
        ("Avg Similarity", metrics_a.avg_similarity, metrics_b.avg_similarity, False),
        ("Cost (USD)", metrics_a.total_cost_usd, metrics_b.total_cost_usd, False),
    ]

    for name, val_a, val_b, is_pct in rows:
        delta = val_b - val_a
        if is_pct:
            fmt = ".1%"
            delta_str = f"{delta:+.1%}"
        else:
            fmt = ".4f"
            delta_str = f"{delta:+.4f}"
        table.add_row(name, f"{val_a:{fmt}}", f"{val_b:{fmt}}", delta_str)

    console.print(table)


def write_eval_json(
    metrics: EvalMetrics,
    case_results: list[CaseResult],
    output_path: Path,
) -> None:
    """Write evaluation results to a JSON file."""
    data = {
        "metrics": asdict(metrics),
        "cases": [
            {
                "case_id": cr.case_id,
                "generated_count": len(cr.generated_cards),
                "matched_count": sum(
                    1 for m in cr.matches if m.matched_card is not None
                ),
                "unmatched_count": len(cr.unmatched_generated),
            }
            for cr in case_results
        ],
    }
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
