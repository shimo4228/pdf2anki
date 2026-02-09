"""Tests for pdf2anki.cost - TDD RED phase.

Tests cover:
- MODEL_PRICING constant (known models and rates)
- CostRecord dataclass (immutable cost record)
- CostTracker (immutable accumulator with budget checking)
- estimate_cost() utility function
- select_model() model routing (Haiku vs Sonnet)
"""

from __future__ import annotations

import pytest

from pdf2anki.cost import (
    MODEL_PRICING,
    CostRecord,
    CostTracker,
    estimate_cost,
    select_model,
)

# ============================================================
# MODEL_PRICING Tests
# ============================================================


class TestModelPricing:
    """Test MODEL_PRICING constant."""

    def test_contains_sonnet(self) -> None:
        matching = [k for k in MODEL_PRICING if "sonnet" in k]
        assert len(matching) >= 1

    def test_contains_haiku(self) -> None:
        matching = [k for k in MODEL_PRICING if "haiku" in k]
        assert len(matching) >= 1

    def test_pricing_has_input_and_output(self) -> None:
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing, f"{model} missing input price"
            assert "output" in pricing, f"{model} missing output price"
            assert pricing["input"] > 0
            assert pricing["output"] > 0

    def test_haiku_cheaper_than_sonnet(self) -> None:
        haiku_key = next(k for k in MODEL_PRICING if "haiku" in k)
        sonnet_key = next(k for k in MODEL_PRICING if "sonnet" in k)
        assert MODEL_PRICING[haiku_key]["input"] < MODEL_PRICING[sonnet_key]["input"]


# ============================================================
# CostRecord Tests
# ============================================================


class TestCostRecord:
    """Test CostRecord immutable dataclass."""

    def test_create_record(self) -> None:
        record = CostRecord(
            model="claude-sonnet-4-5-20250929",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.0105,
        )
        assert record.model == "claude-sonnet-4-5-20250929"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.cost_usd == pytest.approx(0.0105)

    def test_frozen_immutability(self) -> None:
        record = CostRecord(
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        with pytest.raises(AttributeError):
            record.cost_usd = 999.0  # type: ignore[misc]

    def test_zero_tokens_allowed(self) -> None:
        record = CostRecord(
            model="test",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
        assert record.cost_usd == 0.0


# ============================================================
# CostTracker Tests
# ============================================================


class TestCostTracker:
    """Test CostTracker immutable accumulator."""

    def test_empty_tracker(self) -> None:
        tracker = CostTracker(budget_limit=1.00)
        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0
        assert tracker.is_within_budget is True

    def test_add_record(self) -> None:
        tracker = CostTracker(budget_limit=1.00)
        record = CostRecord(
            model="test", input_tokens=100, output_tokens=50, cost_usd=0.01
        )
        new_tracker = tracker.add(record)
        assert new_tracker.total_cost == pytest.approx(0.01)
        assert new_tracker.request_count == 1

    def test_add_returns_new_instance(self) -> None:
        """add() must return a new CostTracker, not mutate the original."""
        tracker = CostTracker(budget_limit=1.00)
        record = CostRecord(
            model="test", input_tokens=100, output_tokens=50, cost_usd=0.01
        )
        new_tracker = tracker.add(record)
        # Original unchanged
        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0
        # New tracker has the record
        assert new_tracker.total_cost == pytest.approx(0.01)

    def test_multiple_adds(self) -> None:
        tracker = CostTracker(budget_limit=1.00)
        record1 = CostRecord(
            model="test", input_tokens=100, output_tokens=50, cost_usd=0.01
        )
        record2 = CostRecord(
            model="test", input_tokens=200, output_tokens=100, cost_usd=0.02
        )
        tracker = tracker.add(record1).add(record2)
        assert tracker.total_cost == pytest.approx(0.03)
        assert tracker.request_count == 2

    def test_budget_exceeded(self) -> None:
        tracker = CostTracker(budget_limit=0.05)
        record = CostRecord(
            model="test", input_tokens=1000, output_tokens=500, cost_usd=0.06
        )
        tracker = tracker.add(record)
        assert tracker.is_within_budget is False

    def test_budget_exactly_at_limit(self) -> None:
        tracker = CostTracker(budget_limit=0.10)
        record = CostRecord(
            model="test", input_tokens=1000, output_tokens=500, cost_usd=0.10
        )
        tracker = tracker.add(record)
        assert tracker.is_within_budget is True

    def test_records_preserved(self) -> None:
        tracker = CostTracker(budget_limit=1.00)
        record1 = CostRecord(
            model="model-a", input_tokens=100, output_tokens=50, cost_usd=0.01
        )
        record2 = CostRecord(
            model="model-b", input_tokens=200, output_tokens=100, cost_usd=0.02
        )
        tracker = tracker.add(record1).add(record2)
        assert len(tracker.records) == 2
        assert tracker.records[0].model == "model-a"
        assert tracker.records[1].model == "model-b"

    def test_budget_remaining(self) -> None:
        tracker = CostTracker(budget_limit=1.00)
        record = CostRecord(
            model="test", input_tokens=100, output_tokens=50, cost_usd=0.30
        )
        tracker = tracker.add(record)
        assert tracker.budget_remaining == pytest.approx(0.70)

    def test_frozen_immutability(self) -> None:
        tracker = CostTracker(budget_limit=1.00)
        with pytest.raises(AttributeError):
            tracker.budget_limit = 999.0  # type: ignore[misc]

    def test_default_budget_limit(self) -> None:
        tracker = CostTracker()
        assert tracker.budget_limit == 1.00


# ============================================================
# estimate_cost Tests
# ============================================================


class TestEstimateCost:
    """Test estimate_cost utility."""

    def test_known_model_cost(self) -> None:
        cost = estimate_cost(
            model="claude-sonnet-4-5-20250929",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        assert cost > 0

    def test_output_tokens_cost(self) -> None:
        cost_input_only = estimate_cost(
            model="claude-sonnet-4-5-20250929",
            input_tokens=1000,
            output_tokens=0,
        )
        cost_with_output = estimate_cost(
            model="claude-sonnet-4-5-20250929",
            input_tokens=1000,
            output_tokens=1000,
        )
        assert cost_with_output > cost_input_only

    def test_zero_tokens_zero_cost(self) -> None:
        cost = estimate_cost(
            model="claude-sonnet-4-5-20250929",
            input_tokens=0,
            output_tokens=0,
        )
        assert cost == 0.0

    def test_unknown_model_uses_fallback(self) -> None:
        """Unknown model should use most expensive pricing as fallback."""
        cost = estimate_cost(
            model="unknown-model-xyz",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost > 0

    def test_haiku_cheaper_than_sonnet(self) -> None:
        haiku_cost = estimate_cost(
            model="claude-haiku-4-5-20251001",
            input_tokens=10000,
            output_tokens=5000,
        )
        sonnet_cost = estimate_cost(
            model="claude-sonnet-4-5-20250929",
            input_tokens=10000,
            output_tokens=5000,
        )
        assert haiku_cost < sonnet_cost


# ============================================================
# select_model Tests
# ============================================================


class TestSelectModel:
    """Test model routing (Haiku/Sonnet selection)."""

    def test_short_text_uses_haiku(self) -> None:
        """Short, simple text should route to cheaper Haiku."""
        model = select_model(text_length=500, card_count=10)
        assert "haiku" in model

    def test_long_text_uses_sonnet(self) -> None:
        """Long, complex text should route to Sonnet."""
        model = select_model(text_length=50000, card_count=50)
        assert "sonnet" in model

    def test_force_model_override(self) -> None:
        """force_model should override automatic selection."""
        model = select_model(
            text_length=500,
            card_count=10,
            force_model="claude-opus-4-6",
        )
        assert model == "claude-opus-4-6"

    def test_returns_valid_model_id(self) -> None:
        """Should return a full Claude model ID."""
        model = select_model(text_length=1000, card_count=20)
        assert model.startswith("claude-")


# ============================================================
# Batch Pricing Tests (Phase 3)
# ============================================================


class TestBatchPricing:
    """Test batch pricing at 50% discount."""

    def test_batch_pricing_exists(self) -> None:
        from pdf2anki.cost import BATCH_PRICING

        assert len(BATCH_PRICING) > 0

    def test_batch_pricing_is_half_of_standard(self) -> None:
        from pdf2anki.cost import BATCH_PRICING

        for model_id, batch_prices in BATCH_PRICING.items():
            std_prices = MODEL_PRICING.get(model_id)
            if std_prices is not None:
                assert batch_prices["input"] == pytest.approx(
                    std_prices["input"] * 0.5
                )
                assert batch_prices["output"] == pytest.approx(
                    std_prices["output"] * 0.5
                )

    def test_estimate_cost_batch_mode(self) -> None:
        """estimate_cost(batch=True) should use 50% pricing."""
        standard = estimate_cost(
            model="claude-sonnet-4-5-20250929",
            input_tokens=10_000,
            output_tokens=5_000,
        )
        batch = estimate_cost(
            model="claude-sonnet-4-5-20250929",
            input_tokens=10_000,
            output_tokens=5_000,
            batch=True,
        )
        assert batch == pytest.approx(standard * 0.5)

    def test_estimate_cost_batch_default_false(self) -> None:
        """Default batch=False should use standard pricing."""
        cost = estimate_cost(
            model="claude-sonnet-4-5-20250929",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        assert cost == pytest.approx(3.00)  # $3/1M input tokens

    def test_estimate_cost_batch_unknown_model_fallback(self) -> None:
        """Unknown model with batch=True should use fallback * 0.5."""
        cost = estimate_cost(
            model="unknown-model",
            input_tokens=1_000_000,
            output_tokens=0,
            batch=True,
        )
        # Fallback input: $15/1M, batch: $7.50/1M
        assert cost == pytest.approx(7.50)
