"""Immutable state models and pure helper functions for the review TUI.

All state transitions return new instances — no mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from pdf2anki.schemas import AnkiCard, CardConfidenceScore


class CardStatus(StrEnum):
    """Review status for a single card."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ReviewCard:
    """A single card under review. Immutable."""

    card: AnkiCard
    original_index: int
    status: CardStatus = CardStatus.PENDING
    quality_score: CardConfidenceScore | None = None


@dataclass(frozen=True)
class ReviewState:
    """Entire review session state. Immutable."""

    items: tuple[ReviewCard, ...]
    current_index: int = 0
    filter_status: CardStatus | None = None

    @property
    def stats(self) -> dict[str, int]:
        """Return counts by status."""
        return {
            "total": len(self.items),
            "accepted": sum(1 for i in self.items if i.status == CardStatus.ACCEPTED),
            "rejected": sum(1 for i in self.items if i.status == CardStatus.REJECTED),
            "pending": sum(1 for i in self.items if i.status == CardStatus.PENDING),
        }

    def filtered_items(self) -> list[ReviewCard]:
        """Return items matching the current filter (or all if None)."""
        if self.filter_status is None:
            return list(self.items)
        return [i for i in self.items if i.status == self.filter_status]


# ── Pure helper functions ─────────────────────────────────────


def create_initial_state(
    cards: list[AnkiCard],
    scores: list[CardConfidenceScore] | None = None,
) -> ReviewState:
    """Build initial ReviewState from a card list and optional scores."""
    items = tuple(
        ReviewCard(
            card=card,
            original_index=idx,
            quality_score=scores[idx] if scores else None,
        )
        for idx, card in enumerate(cards)
    )
    return ReviewState(items=items)


def set_card_status(
    state: ReviewState, item_index: int, status: CardStatus
) -> ReviewState:
    """Return new state with the specified card's status changed."""
    items = list(state.items)
    items[item_index] = replace(items[item_index], status=status)
    return replace(state, items=tuple(items))


def edit_card(
    state: ReviewState, item_index: int, front: str, back: str
) -> ReviewState:
    """Return new state with the specified card's front/back edited."""
    items = list(state.items)
    old_item = items[item_index]
    new_card = old_item.card.model_copy(update={"front": front, "back": back})
    items[item_index] = replace(old_item, card=new_card)
    return replace(state, items=tuple(items))


def navigate(state: ReviewState, delta: int) -> ReviewState:
    """Return new state with current_index moved by delta (wrapping)."""
    count = len(state.items)
    if count == 0:
        return state
    new_index = (state.current_index + delta) % count
    return replace(state, current_index=new_index)


_FILTER_CYCLE: list[CardStatus | None] = [
    None,
    CardStatus.PENDING,
    CardStatus.ACCEPTED,
    CardStatus.REJECTED,
]


def cycle_filter(state: ReviewState) -> ReviewState:
    """Cycle filter: None → pending → accepted → rejected → None."""
    current_pos = _FILTER_CYCLE.index(state.filter_status)
    next_pos = (current_pos + 1) % len(_FILTER_CYCLE)
    return replace(state, filter_status=_FILTER_CYCLE[next_pos])
