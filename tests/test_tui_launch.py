"""Tests for pdf2anki.tui.__init__ (launch_review function)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pdf2anki.schemas import AnkiCard, BloomLevel, CardType, ExtractionResult

_CARD = AnkiCard(
    front="What is X?",
    back="X is Y.",
    card_type=CardType.QA,
    bloom_level=BloomLevel.UNDERSTAND,
    tags=["test"],
)

_RESULT = ExtractionResult(
    source_file="test.pdf",
    cards=[_CARD],
    model_used="claude-sonnet-4-5-20250929",
)


class TestLaunchReview:
    """Tests for launch_review in tui/__init__.py."""

    @patch("pdf2anki.tui.ReviewApp")
    @patch("pdf2anki.tui.create_initial_state")
    def test_quit_returns_original(
        self, mock_state: MagicMock, mock_app_cls: MagicMock
    ) -> None:
        """When user quits (no save), original result is returned."""
        mock_state.return_value = MagicMock()
        mock_app = MagicMock()
        mock_app.save_requested = False
        mock_app_cls.return_value = mock_app

        from pdf2anki.tui import launch_review

        result = launch_review(_RESULT)
        assert result is _RESULT
        mock_app.run.assert_called_once()

    @patch("pdf2anki.tui.ReviewApp")
    @patch("pdf2anki.tui.create_initial_state")
    def test_save_returns_accepted_cards(
        self, mock_state: MagicMock, mock_app_cls: MagicMock
    ) -> None:
        """When user saves, only accepted cards are returned."""
        from pdf2anki.tui.state import CardStatus, ReviewCard

        item = ReviewCard(card=_CARD, original_index=0, status=CardStatus.ACCEPTED)
        state = MagicMock()
        state.items = [item]
        mock_state.return_value = state

        mock_app = MagicMock()
        mock_app.save_requested = True
        mock_app.state = state
        mock_app_cls.return_value = mock_app

        from pdf2anki.tui import launch_review

        result = launch_review(_RESULT)
        assert len(result.cards) == 1
        assert result.cards[0] == _CARD

    @patch("pdf2anki.tui.ReviewApp")
    @patch("pdf2anki.tui.create_initial_state")
    def test_save_with_rejected_cards(
        self, mock_state: MagicMock, mock_app_cls: MagicMock
    ) -> None:
        """When user saves with rejected cards, they are filtered out."""
        from pdf2anki.tui.state import CardStatus, ReviewCard

        item = ReviewCard(card=_CARD, original_index=0, status=CardStatus.REJECTED)
        state = MagicMock()
        state.items = [item]
        mock_state.return_value = state

        mock_app = MagicMock()
        mock_app.save_requested = True
        mock_app.state = state
        mock_app_cls.return_value = mock_app

        from pdf2anki.tui import launch_review

        result = launch_review(_RESULT)
        assert len(result.cards) == 0
