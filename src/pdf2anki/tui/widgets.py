"""Custom widgets for the review TUI: StatsBar, CardDisplay, ActionBar."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from pdf2anki.tui.state import CardStatus, ReviewCard


class StatsBar(Static):
    """Top bar showing total/accepted/rejected/pending counts."""

    def update_stats(self, stats: dict[str, int]) -> None:
        """Refresh the displayed statistics."""
        text = (
            f" Total: {stats['total']}"
            f" | Accepted: {stats['accepted']}"
            f" | Rejected: {stats['rejected']}"
            f" | Pending: {stats['pending']}"
        )
        self.update(text)


class CardDisplay(Container):
    """Central area displaying a single card's details."""

    DEFAULT_CSS = """
    CardDisplay {
        height: 1fr;
        padding: 1 2;
    }
    #card-content {
        width: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="card-content")

    def render_card_text(
        self, item: ReviewCard, index: int, total: int
    ) -> str:
        """Build the display text for a card (pure function)."""
        card = item.card
        status_label = item.status.value.upper()
        quality = ""
        if item.quality_score is not None:
            quality = f"  Quality: {item.quality_score.weighted_total:.2f}"

        lines = [
            f"Card {index + 1}/{total}  [{card.card_type.value.upper()}]"
            f"  Status: {status_label}{quality}",
            "",
            f"Front: {card.front}",
            "─" * 60,
            f"Back:  {card.back}",
            "─" * 60,
            f"Tags:  {', '.join(card.tags)}",
            f"Bloom: {card.bloom_level.value}",
        ]
        if card.related_concepts:
            lines.append(f"Related: {', '.join(card.related_concepts)}")
        if card.mnemonic_hint:
            lines.append(f"Hint: {card.mnemonic_hint}")

        return "\n".join(lines)

    def update_card(self, item: ReviewCard, index: int, total: int) -> None:
        """Update the display with a new card."""
        text = self.render_card_text(item, index, total)
        content = self.query_one("#card-content", Static)
        content.update(text)


_STATUS_LABELS = {
    None: "All",
    CardStatus.PENDING: "Pending",
    CardStatus.ACCEPTED: "Accepted",
    CardStatus.REJECTED: "Rejected",
}


class ActionBar(Static):
    """Bottom bar showing key bindings and filter status."""

    def update_actions(self, filter_status: CardStatus | None) -> None:
        """Refresh the action bar text."""
        filter_label = _STATUS_LABELS.get(filter_status, "All")
        text = (
            " [A]ccept  [R]eject  [E]dit  [N]ext  [P]rev"
            f"  [F]ilter({filter_label})  [S]ave & Quit  [Q]uit"
        )
        self.update(text)
