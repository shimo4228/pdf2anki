"""Tests for tui/widgets.py and tui/app.py using Textual Pilot API."""

from __future__ import annotations

import pytest

from pdf2anki.schemas import AnkiCard, BloomLevel, CardType
from pdf2anki.tui.app import EditCardScreen, ReviewApp
from pdf2anki.tui.state import (
    CardStatus,
    ReviewState,
    create_initial_state,
)
from pdf2anki.tui.widgets import ActionBar, CardDisplay, StatsBar


# ── helpers ───────────────────────────────────────────────────


def _make_cards(n: int = 5) -> list[AnkiCard]:
    return [
        AnkiCard(
            front=f"Question {i}?",
            back=f"Answer {i}",
            card_type=CardType.QA,
            bloom_level=BloomLevel.UNDERSTAND,
            tags=["test::tag"],
        )
        for i in range(n)
    ]


def _sample_state(n: int = 5) -> ReviewState:
    return create_initial_state(_make_cards(n))


# ── Widget rendering ──────────────────────────────────────────


class TestStatsBar:
    @pytest.mark.asyncio
    async def test_renders_counts(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            bar = pilot.app.query_one(StatsBar)
            assert "3" in bar.content  # total count


class TestCardDisplay:
    @pytest.mark.asyncio
    async def test_shows_card_content(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            display = pilot.app.query_one(CardDisplay)
            text = display.render_card_text(
                pilot.app.state.items[0], 0, 3
            )
            assert "Question 0?" in text
            assert "Answer 0" in text


class TestActionBar:
    @pytest.mark.asyncio
    async def test_renders(self) -> None:
        app = ReviewApp(_sample_state())
        async with app.run_test() as pilot:
            bar = pilot.app.query_one(ActionBar)
            assert bar is not None


# ── ReviewApp key bindings ────────────────────────────────────


class TestReviewAppAccept:
    @pytest.mark.asyncio
    async def test_accept_sets_status(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            await pilot.press("a")
            assert app.state.items[0].status == CardStatus.ACCEPTED

    @pytest.mark.asyncio
    async def test_accept_advances(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            await pilot.press("a")
            assert app.state.current_index == 1


class TestReviewAppReject:
    @pytest.mark.asyncio
    async def test_reject_sets_status(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            await pilot.press("r")
            assert app.state.items[0].status == CardStatus.REJECTED

    @pytest.mark.asyncio
    async def test_reject_advances(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            await pilot.press("r")
            assert app.state.current_index == 1


class TestReviewAppNavigation:
    @pytest.mark.asyncio
    async def test_next(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            await pilot.press("n")
            assert app.state.current_index == 1

    @pytest.mark.asyncio
    async def test_prev_wraps(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            await pilot.press("p")
            assert app.state.current_index == 2


class TestReviewAppFilter:
    @pytest.mark.asyncio
    async def test_cycle_filter(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            await pilot.press("f")
            assert app.state.filter_status == CardStatus.PENDING
            await pilot.press("f")
            assert app.state.filter_status == CardStatus.ACCEPTED


class TestReviewAppSaveQuit:
    @pytest.mark.asyncio
    async def test_save_sets_flag(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            await pilot.press("s")
            assert app.save_requested is True

    @pytest.mark.asyncio
    async def test_quit_no_save(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            await pilot.press("q")
            assert app.save_requested is False


class TestEditCardScreen:
    @pytest.mark.asyncio
    async def test_edit_opens_modal(self) -> None:
        app = ReviewApp(_sample_state(3))
        async with app.run_test() as pilot:
            await pilot.press("e")
            assert isinstance(app.screen, EditCardScreen)
