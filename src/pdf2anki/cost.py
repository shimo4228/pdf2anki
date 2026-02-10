"""Cost tracking and model routing for pdf2anki.

Provides immutable CostTracker (g-kentei-ios pattern), cost estimation
for Claude models, and automatic model selection (Haiku/Sonnet routing).
"""

from __future__ import annotations

from dataclasses import dataclass

# Model ID constants
MODEL_SONNET = "claude-sonnet-4-5-20250929"
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_OPUS = "claude-opus-4-6"

# Pricing per 1M tokens (USD) as of 2025-2026
MODEL_PRICING: dict[str, dict[str, float]] = {
    MODEL_SONNET: {"input": 3.00, "output": 15.00},
    MODEL_HAIKU: {"input": 0.80, "output": 4.00},
    MODEL_OPUS: {"input": 15.00, "output": 75.00},
}

# Batch API pricing (50% of standard)
BATCH_PRICING: dict[str, dict[str, float]] = {
    model_id: {"input": p["input"] * 0.5, "output": p["output"] * 0.5}
    for model_id, p in MODEL_PRICING.items()
}

# Fallback pricing (most expensive to avoid underestimation)
_FALLBACK_PRICING = {"input": 15.00, "output": 75.00}

# Threshold for routing to Sonnet (chars)
_SONNET_TEXT_THRESHOLD = 10_000
_SONNET_CARD_THRESHOLD = 30


@dataclass(frozen=True, slots=True)
class CostRecord:
    """A single API call cost record. Immutable."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class CostTracker:
    """Immutable cost tracker with budget enforcement.

    Use add() to create a new tracker with an additional record.
    Original instance is never mutated (g-kentei-ios pattern).
    """

    budget_limit: float = 1.00
    records: tuple[CostRecord, ...] = ()

    @property
    def total_cost(self) -> float:
        """Sum of all recorded costs."""
        return sum(r.cost_usd for r in self.records)

    @property
    def request_count(self) -> int:
        """Number of recorded API calls."""
        return len(self.records)

    @property
    def is_within_budget(self) -> bool:
        """True if total cost is at or below budget limit."""
        return self.total_cost <= self.budget_limit

    @property
    def budget_remaining(self) -> float:
        """Remaining budget in USD."""
        return self.budget_limit - self.total_cost

    def add(self, record: CostRecord) -> CostTracker:
        """Return a new CostTracker with the record appended."""
        return CostTracker(
            budget_limit=self.budget_limit,
            records=(*self.records, record),
        )


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    batch: bool = False,
) -> float:
    """Estimate the cost of an API call in USD.

    Args:
        model: Claude model ID.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        batch: If True, use batch pricing (50% of standard).

    Returns:
        Estimated cost in USD.
    """
    if batch:
        fallback = {"input": _FALLBACK_PRICING["input"] * 0.5,
                     "output": _FALLBACK_PRICING["output"] * 0.5}
        pricing = BATCH_PRICING.get(model, fallback)
    else:
        pricing = MODEL_PRICING.get(model, _FALLBACK_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate Claude Vision API input tokens for an image.

    Formula: tokens = (width * height) / 750
    Reference: 1092x1092 ~ 1,590 tokens
    """
    return (width * height) // 750


def select_model(
    text_length: int,
    card_count: int,
    *,
    force_model: str | None = None,
) -> str:
    """Select the optimal Claude model based on task complexity.

    Routes to Haiku for simple tasks (short text, few cards) and
    Sonnet for complex tasks (long text, many cards).

    Args:
        text_length: Length of source text in characters.
        card_count: Requested number of cards.
        force_model: Override automatic selection with this model.

    Returns:
        Claude model ID string.
    """
    if force_model is not None:
        return force_model

    if text_length >= _SONNET_TEXT_THRESHOLD or card_count >= _SONNET_CARD_THRESHOLD:
        return MODEL_SONNET

    return MODEL_HAIKU
