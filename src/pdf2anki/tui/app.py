"""ReviewApp and EditCardScreen for the interactive card review TUI."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from pdf2anki.tui.state import (
    CardStatus,
    ReviewState,
    cycle_filter,
    edit_card,
    navigate,
    set_card_status,
)
from pdf2anki.tui.widgets import ActionBar, CardDisplay, StatsBar


class EditCardScreen(ModalScreen[tuple[str, str] | None]):
    """Modal screen for editing a card's front/back text."""

    DEFAULT_CSS = """
    EditCardScreen {
        align: center middle;
    }
    #edit-dialog {
        width: 80;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }
    #edit-front, #edit-back {
        height: 5;
        margin-bottom: 1;
    }
    #edit-buttons {
        height: 3;
        align-horizontal: center;
    }
    """

    def __init__(self, front: str, back: str) -> None:
        self._initial_front = front
        self._initial_back = back
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-dialog"):
            yield Static("Edit Card", classes="title")
            yield Static("Front:")
            yield TextArea(self._initial_front, id="edit-front")
            yield Static("Back:")
            yield TextArea(self._initial_back, id="edit-back")
            yield Button("Save", id="edit-save", variant="primary")
            yield Button("Cancel", id="edit-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle save/cancel button clicks."""
        if event.button.id == "edit-save":
            front_area = self.query_one("#edit-front", TextArea)
            back_area = self.query_one("#edit-back", TextArea)
            self.dismiss((front_area.text, back_area.text))
        else:
            self.dismiss(None)


class ReviewApp(App[None]):
    """Main TUI application for reviewing generated Anki cards."""

    TITLE = "pdf2anki Review"

    DEFAULT_CSS = """
    Screen {
        layout: vertical;
    }
    StatsBar {
        dock: top;
        height: 1;
        background: $accent;
        color: $text;
        padding: 0 1;
    }
    ActionBar {
        dock: bottom;
        height: 1;
        background: $accent;
        color: $text;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("a", "accept_card", "Accept"),
        Binding("r", "reject_card", "Reject"),
        Binding("e", "edit_card", "Edit"),
        Binding("n", "next_card", "Next"),
        Binding("p", "prev_card", "Previous"),
        Binding("f", "cycle_filter", "Filter"),
        Binding("s", "save_and_quit", "Save & Quit"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self, initial_state: ReviewState) -> None:
        self.state = initial_state
        self.save_requested = False
        super().__init__()

    def compose(self) -> ComposeResult:
        yield StatsBar("")
        yield CardDisplay()
        yield ActionBar("")

    def on_mount(self) -> None:
        """Initialize the UI after mounting."""
        self._refresh_ui()

    # ── Actions ───────────────────────────────────────────

    def action_accept_card(self) -> None:
        """Accept the current card and advance."""
        idx = self._current_item_index()
        if idx is not None:
            self.state = set_card_status(self.state, idx, CardStatus.ACCEPTED)
            self.state = navigate(self.state, +1)
            self._refresh_ui()

    def action_reject_card(self) -> None:
        """Reject the current card and advance."""
        idx = self._current_item_index()
        if idx is not None:
            self.state = set_card_status(self.state, idx, CardStatus.REJECTED)
            self.state = navigate(self.state, +1)
            self._refresh_ui()

    def action_edit_card(self) -> None:
        """Open the edit modal for the current card."""
        idx = self._current_item_index()
        if idx is not None:
            item = self.state.items[idx]
            self.push_screen(
                EditCardScreen(item.card.front, item.card.back),
                callback=self._on_edit_complete,
            )

    def action_next_card(self) -> None:
        """Navigate to the next card."""
        self.state = navigate(self.state, +1)
        self._refresh_ui()

    def action_prev_card(self) -> None:
        """Navigate to the previous card."""
        self.state = navigate(self.state, -1)
        self._refresh_ui()

    def action_cycle_filter(self) -> None:
        """Cycle the display filter."""
        self.state = cycle_filter(self.state)
        self._refresh_ui()

    def action_save_and_quit(self) -> None:
        """Save accepted cards and exit."""
        self.save_requested = True
        self.exit()

    def action_quit_app(self) -> None:
        """Quit without saving."""
        self.save_requested = False
        self.exit()

    # ── Internal ──────────────────────────────────────────

    def _current_item_index(self) -> int | None:
        """Return the real item index for the current position."""
        filtered = self.state.filtered_items()
        if not filtered:
            return None
        pos = self.state.current_index % len(filtered)
        item = filtered[pos]
        return item.original_index

    def _on_edit_complete(self, result: tuple[str, str] | None) -> None:
        """Handle edit modal result."""
        if result is not None:
            idx = self._current_item_index()
            if idx is not None:
                new_front, new_back = result
                self.state = edit_card(self.state, idx, new_front, new_back)
                self._refresh_ui()

    def _refresh_ui(self) -> None:
        """Update all widgets to reflect the current state."""
        stats_bar = self.query_one(StatsBar)
        stats_bar.update_stats(self.state.stats)

        action_bar = self.query_one(ActionBar)
        action_bar.update_actions(self.state.filter_status)

        display = self.query_one(CardDisplay)
        filtered = self.state.filtered_items()
        if filtered:
            pos = self.state.current_index % len(filtered)
            display.update_card(filtered[pos], pos, len(filtered))
        else:
            content = display.query_one("#card-content", Static)
            content.update("No cards match the current filter.")
