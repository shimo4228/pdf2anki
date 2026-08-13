"""Cost tracking and model routing for pdf2anki.

Provides immutable CostTracker (g-kentei-ios pattern), cost estimation
for Claude models, and automatic model selection (Haiku/Sonnet routing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Model ID constants
MODEL_SONNET = "claude-sonnet-4-5-20250929"
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_OPUS = "claude-opus-4-6"

# Pricing per 1M tokens (USD) as of 2026-08 (platform.claude.com/docs/en/pricing)
MODEL_PRICING: dict[str, dict[str, float]] = {
    MODEL_SONNET: {"input": 3.00, "output": 15.00},
    MODEL_HAIKU: {"input": 1.00, "output": 5.00},
    MODEL_OPUS: {"input": 5.00, "output": 25.00},
}

# Batch API pricing (50% of standard)
BATCH_PRICING: dict[str, dict[str, float]] = {
    model_id: {"input": p["input"] * 0.5, "output": p["output"] * 0.5}
    for model_id, p in MODEL_PRICING.items()
}

# Fallback pricing (highest current Claude pricing to avoid underestimation)
_FALLBACK_PRICING = {"input": 10.00, "output": 50.00}

# Prompt-cache multipliers relative to the input price:
# cache writes bill at 1.25x, cache reads at 0.1x.
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.1

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
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


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
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate the cost of an API call in USD.

    Args:
        model: Claude model ID.
        input_tokens: Number of uncached input tokens.
        output_tokens: Number of output tokens.
        batch: If True, use batch pricing (50% of standard).
        cache_creation_tokens: Prompt-cache write tokens (billed at 1.25x input).
        cache_read_tokens: Prompt-cache read tokens (billed at 0.1x input).

    Returns:
        Estimated cost in USD.
    """
    if batch:
        fallback = {
            "input": _FALLBACK_PRICING["input"] * 0.5,
            "output": _FALLBACK_PRICING["output"] * 0.5,
        }
        pricing = BATCH_PRICING.get(model, fallback)
    else:
        pricing = MODEL_PRICING.get(model, _FALLBACK_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    cache_cost = (cache_creation_tokens / 1_000_000) * pricing[
        "input"
    ] * _CACHE_WRITE_MULTIPLIER + (cache_read_tokens / 1_000_000) * pricing[
        "input"
    ] * _CACHE_READ_MULTIPLIER
    return input_cost + output_cost + cache_cost


def _token_count(value: Any) -> int:
    """Coerce a usage field to int; None or absent fields count as 0."""
    return value if isinstance(value, int) else 0


def record_response_cost(
    tracker: CostTracker,
    response: Any,
    *,
    batch: bool = False,
) -> CostTracker:
    """Record an Anthropic API response's full billed usage on the tracker.

    Reads cache_creation_input_tokens / cache_read_input_tokens off the
    usage object when present, so prompt-caching costs are not lost.
    Call this immediately after the API returns, before any content
    validation, so failed parses still account for billed spend.
    """
    usage = response.usage
    cache_creation = _token_count(getattr(usage, "cache_creation_input_tokens", 0))
    cache_read = _token_count(getattr(usage, "cache_read_input_tokens", 0))
    cost = estimate_cost(
        model=response.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        batch=batch,
        cache_creation_tokens=cache_creation,
        cache_read_tokens=cache_read,
    )
    record = CostRecord(
        model=response.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost,
        cache_creation_tokens=cache_creation,
        cache_read_tokens=cache_read,
    )
    return tracker.add(record)


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
