"""Tests for tui/state.py — ReviewState, CardStatus, pure helper functions."""

from __future__ import annotations

import pytest

from pdf2anki.schemas import AnkiCard, BloomLevel, CardConfidenceScore, CardType
from pdf2anki.tui.state import (
    CardStatus,
    ReviewCard,
    ReviewState,
    create_initial_state,
    cycle_filter,
    edit_card,
    navigate,
    set_card_status,
)


# ── fixtures ──────────────────────────────────────────────────


def _make_card(front: str = "Q?", back: str = "A") -> AnkiCard:
    return AnkiCard(
        front=front,
        back=back,
        card_type=CardType.QA,
        bloom_level=BloomLevel.UNDERSTAND,
        tags=["test"],
    )


def _make_score() -> CardConfidenceScore:
    return CardConfidenceScore(
        front_quality=0.9,
        back_quality=0.8,
        card_type_fit=0.9,
        bloom_level_fit=0.9,
        tags_quality=0.7,
        atomicity=1.0,
    )


@pytest.fixture
def three_cards() -> list[AnkiCard]:
    return [_make_card(f"Q{i}?", f"A{i}") for i in range(3)]


@pytest.fixture
def initial_state(three_cards: list[AnkiCard]) -> ReviewState:
    return create_initial_state(three_cards)


# ── CardStatus ────────────────────────────────────────────────


class TestCardStatus:
    def test_three_states_exist(self) -> None:
        assert CardStatus.PENDING == "pending"
        assert CardStatus.ACCEPTED == "accepted"
        assert CardStatus.REJECTED == "rejected"

    def test_all_values(self) -> None:
        assert len(CardStatus) == 3


# ── ReviewCard ────────────────────────────────────────────────


class TestReviewCard:
    def test_frozen(self) -> None:
        rc = ReviewCard(card=_make_card(), original_index=0)
        with pytest.raises(AttributeError):
            rc.status = CardStatus.ACCEPTED  # type: ignore[misc]

    def test_defaults(self) -> None:
        rc = ReviewCard(card=_make_card(), original_index=0)
        assert rc.status == CardStatus.PENDING
        assert rc.quality_score is None

    def test_with_score(self) -> None:
        score = _make_score()
        rc = ReviewCard(card=_make_card(), original_index=0, quality_score=score)
        assert rc.quality_score is not None
        assert rc.quality_score.weighted_total > 0


# ── ReviewState.stats ─────────────────────────────────────────


class TestReviewStateStats:
    def test_all_pending(self, initial_state: ReviewState) -> None:
        stats = initial_state.stats
        assert stats["total"] == 3
        assert stats["pending"] == 3
        assert stats["accepted"] == 0
        assert stats["rejected"] == 0

    def test_mixed_status(self, initial_state: ReviewState) -> None:
        s = set_card_status(initial_state, 0, CardStatus.ACCEPTED)
        s = set_card_status(s, 1, CardStatus.REJECTED)
        stats = s.stats
        assert stats["accepted"] == 1
        assert stats["rejected"] == 1
        assert stats["pending"] == 1


# ── ReviewState.filtered_items ────────────────────────────────


class TestFilteredItems:
    def test_no_filter_returns_all(self, initial_state: ReviewState) -> None:
        assert len(initial_state.filtered_items()) == 3

    def test_filter_pending(self, initial_state: ReviewState) -> None:
        s = set_card_status(initial_state, 0, CardStatus.ACCEPTED)
        filtered_state = ReviewState(
            items=s.items, current_index=0, filter_status=CardStatus.PENDING
        )
        assert len(filtered_state.filtered_items()) == 2

    def test_filter_empty(self, initial_state: ReviewState) -> None:
        filtered_state = ReviewState(
            items=initial_state.items,
            current_index=0,
            filter_status=CardStatus.REJECTED,
        )
        assert len(filtered_state.filtered_items()) == 0


# ── create_initial_state ──────────────────────────────────────


class TestCreateInitialState:
    def test_basic(self, three_cards: list[AnkiCard]) -> None:
        state = create_initial_state(three_cards)
        assert len(state.items) == 3
        assert state.current_index == 0
        assert state.filter_status is None
        for i, item in enumerate(state.items):
            assert item.original_index == i
            assert item.status == CardStatus.PENDING
            assert item.quality_score is None

    def test_with_scores(self, three_cards: list[AnkiCard]) -> None:
        scores = [_make_score() for _ in three_cards]
        state = create_initial_state(three_cards, scores=scores)
        for item in state.items:
            assert item.quality_score is not None

    def test_empty_cards(self) -> None:
        state = create_initial_state([])
        assert len(state.items) == 0
        assert state.current_index == 0


# ── set_card_status ───────────────────────────────────────────


class TestSetCardStatus:
    def test_change_status(self, initial_state: ReviewState) -> None:
        new_state = set_card_status(initial_state, 0, CardStatus.ACCEPTED)
        assert new_state.items[0].status == CardStatus.ACCEPTED

    def test_original_unchanged(self, initial_state: ReviewState) -> None:
        set_card_status(initial_state, 0, CardStatus.ACCEPTED)
        assert initial_state.items[0].status == CardStatus.PENDING

    def test_other_items_unchanged(self, initial_state: ReviewState) -> None:
        new_state = set_card_status(initial_state, 1, CardStatus.REJECTED)
        assert new_state.items[0].status == CardStatus.PENDING
        assert new_state.items[2].status == CardStatus.PENDING


# ── edit_card ─────────────────────────────────────────────────


class TestEditCard:
    def test_edit_front_back(self, initial_state: ReviewState) -> None:
        new_state = edit_card(initial_state, 0, "New Front?", "New Back")
        assert new_state.items[0].card.front == "New Front?"
        assert new_state.items[0].card.back == "New Back"

    def test_original_card_unchanged(self, initial_state: ReviewState) -> None:
        edit_card(initial_state, 0, "Changed?", "Changed")
        assert initial_state.items[0].card.front == "Q0?"

    def test_other_items_unchanged(self, initial_state: ReviewState) -> None:
        new_state = edit_card(initial_state, 0, "X?", "Y")
        assert new_state.items[1].card.front == "Q1?"


# ── navigate ──────────────────────────────────────────────────


class TestNavigate:
    def test_next(self, initial_state: ReviewState) -> None:
        s = navigate(initial_state, +1)
        assert s.current_index == 1

    def test_prev(self, initial_state: ReviewState) -> None:
        s = ReviewState(items=initial_state.items, current_index=1)
        s = navigate(s, -1)
        assert s.current_index == 0

    def test_wrap_forward(self, initial_state: ReviewState) -> None:
        s = ReviewState(items=initial_state.items, current_index=2)
        s = navigate(s, +1)
        assert s.current_index == 0

    def test_wrap_backward(self, initial_state: ReviewState) -> None:
        s = navigate(initial_state, -1)
        assert s.current_index == 2

    def test_empty_state(self) -> None:
        s = create_initial_state([])
        s = navigate(s, +1)
        assert s.current_index == 0


# ── cycle_filter ──────────────────────────────────────────────


class TestCycleFilter:
    def test_full_cycle(self, initial_state: ReviewState) -> None:
        s = initial_state
        assert s.filter_status is None

        s = cycle_filter(s)
        assert s.filter_status == CardStatus.PENDING

        s = cycle_filter(s)
        assert s.filter_status == CardStatus.ACCEPTED

        s = cycle_filter(s)
        assert s.filter_status == CardStatus.REJECTED

        s = cycle_filter(s)
        assert s.filter_status is None
