"""Interactive card review TUI for pdf2anki.

Public API: launch_review() — insert between process_file() and write_output().
"""

from __future__ import annotations

from pdf2anki.schemas import CardConfidenceScore, ExtractionResult
from pdf2anki.tui.app import ReviewApp
from pdf2anki.tui.state import CardStatus, create_initial_state


def launch_review(
    result: ExtractionResult,
    scores: list[CardConfidenceScore] | None = None,
) -> ExtractionResult:
    """Launch the interactive review TUI and return filtered result.

    If the user saves (press 's'), only accepted cards are returned.
    If the user quits (press 'q'), the original result is returned unchanged.
    """
    state = create_initial_state(list(result.cards), scores=scores)
    app = ReviewApp(state)
    app.run()

    if not app.save_requested:
        return result

    accepted_cards = [
        item.card
        for item in app.state.items
        if item.status == CardStatus.ACCEPTED
    ]
    return result.model_copy(update={"cards": accepted_cards})
