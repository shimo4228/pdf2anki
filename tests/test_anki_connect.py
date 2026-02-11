"""Tests for AnkiConnect API integration."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from pdf2anki.schemas import AnkiCard, BloomLevel, CardType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qa_card() -> AnkiCard:
    return AnkiCard(
        front="活性化関数の役割は？",
        back="非線形変換を導入する。",
        card_type=CardType.QA,
        bloom_level=BloomLevel.UNDERSTAND,
        tags=["AI::基礎"],
    )


@pytest.fixture
def cloze_card() -> AnkiCard:
    return AnkiCard(
        front="{{c1::勾配降下法}}は損失関数を最小化する。",
        back="",
        card_type=CardType.CLOZE,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::最適化"],
    )


@pytest.fixture
def reversible_card() -> AnkiCard:
    return AnkiCard(
        front="ReLU",
        back="Rectified Linear Unit: f(x) = max(0, x)",
        card_type=CardType.REVERSIBLE,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::活性化関数"],
    )


@pytest.fixture
def term_card() -> AnkiCard:
    return AnkiCard(
        front="バッチ正規化",
        back="各層の入力を正規化する手法。",
        card_type=CardType.TERM_DEFINITION,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::正規化"],
    )


def _mock_response(result: object = None, error: str | None = None) -> MagicMock:
    """Create a mock urllib response with JSON body."""
    body = json.dumps({"result": result, "error": error}).encode()
    mock = MagicMock()
    mock.read.return_value = body
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


# ---------------------------------------------------------------------------
# _invoke
# ---------------------------------------------------------------------------


class TestInvoke:
    def test_invoke_success(self) -> None:
        from pdf2anki.anki_connect import _invoke

        with patch("pdf2anki.anki_connect.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _mock_response(result=6)
            result = _invoke("version")

        assert result == 6

    def test_invoke_sends_correct_json(self) -> None:
        from pdf2anki.anki_connect import _invoke

        with patch("pdf2anki.anki_connect.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _mock_response(result=None)
            _invoke("createDeck", deck="test")

        call_args = mock_urlopen.call_args[0][0]
        body = json.loads(call_args.data)
        assert body["action"] == "createDeck"
        assert body["version"] == 6
        assert body["params"] == {"deck": "test"}

    def test_invoke_raises_on_api_error(self) -> None:
        from pdf2anki.anki_connect import AnkiConnectError, _invoke

        with patch("pdf2anki.anki_connect.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _mock_response(error="model not found")
            with pytest.raises(AnkiConnectError, match="model not found"):
                _invoke("addNote", note={})

    def test_invoke_raises_on_connection_error(self) -> None:
        from pdf2anki.anki_connect import AnkiConnectError, _invoke

        with patch("pdf2anki.anki_connect.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
            with pytest.raises(AnkiConnectError, match="Anki"):
                _invoke("version")


# ---------------------------------------------------------------------------
# is_anki_running
# ---------------------------------------------------------------------------


class TestIsAnkiRunning:
    def test_returns_true_when_connected(self) -> None:
        from pdf2anki.anki_connect import is_anki_running

        with patch("pdf2anki.anki_connect._invoke", return_value=6):
            assert is_anki_running() is True

    def test_returns_false_on_connection_error(self) -> None:
        from pdf2anki.anki_connect import AnkiConnectError, is_anki_running

        with patch(
            "pdf2anki.anki_connect._invoke",
            side_effect=AnkiConnectError("fail"),
        ):
            assert is_anki_running() is False


# ---------------------------------------------------------------------------
# ensure_deck
# ---------------------------------------------------------------------------


class TestEnsureDeck:
    def test_creates_deck(self) -> None:
        from pdf2anki.anki_connect import ensure_deck

        with patch("pdf2anki.anki_connect._invoke", return_value=123) as mock:
            ensure_deck("pdf2anki::test")

        mock.assert_called_once_with("createDeck", deck="pdf2anki::test")

    def test_propagates_error(self) -> None:
        from pdf2anki.anki_connect import AnkiConnectError, ensure_deck

        with patch(
            "pdf2anki.anki_connect._invoke",
            side_effect=AnkiConnectError("err"),
        ):
            with pytest.raises(AnkiConnectError):
                ensure_deck("bad")


# ---------------------------------------------------------------------------
# card_to_note
# ---------------------------------------------------------------------------


class TestCardToNote:
    def test_qa_card_uses_basic_model(self, qa_card: AnkiCard) -> None:
        from pdf2anki.anki_connect import card_to_note

        note = card_to_note(qa_card, deck_name="test")
        assert note["modelName"] == "Basic"
        assert note["fields"]["Front"] == qa_card.front
        assert note["fields"]["Back"] == qa_card.back
        assert note["deckName"] == "test"

    def test_cloze_card_uses_cloze_model(self, cloze_card: AnkiCard) -> None:
        from pdf2anki.anki_connect import card_to_note

        note = card_to_note(cloze_card, deck_name="test")
        assert note["modelName"] == "Cloze"
        assert "{{c1::" in note["fields"]["Text"]
        assert note["fields"]["Extra"] == ""

    def test_term_card_uses_basic_model(self, term_card: AnkiCard) -> None:
        from pdf2anki.anki_connect import card_to_note

        note = card_to_note(term_card, deck_name="test")
        assert note["modelName"] == "Basic"

    def test_reversible_card_uses_basic_reversed_model(
        self, reversible_card: AnkiCard
    ) -> None:
        from pdf2anki.anki_connect import card_to_note

        note = card_to_note(reversible_card, deck_name="test")
        assert note["modelName"] == "Basic (and target: reversed card)"

    def test_tags_are_included(self, qa_card: AnkiCard) -> None:
        from pdf2anki.anki_connect import card_to_note

        note = card_to_note(qa_card, deck_name="test")
        assert "AI::基礎" in note["tags"]

    def test_bloom_tag_added(self, qa_card: AnkiCard) -> None:
        from pdf2anki.anki_connect import card_to_note

        note = card_to_note(qa_card, deck_name="test")
        assert "bloom::understand" in note["tags"]

    def test_options_no_duplicate(self, qa_card: AnkiCard) -> None:
        from pdf2anki.anki_connect import card_to_note

        note = card_to_note(qa_card, deck_name="test")
        assert note["options"]["allowDuplicate"] is False


# ---------------------------------------------------------------------------
# push_cards
# ---------------------------------------------------------------------------


class TestPushCards:
    def test_push_empty_list(self) -> None:
        from pdf2anki.anki_connect import push_cards

        result = push_cards([], deck_name="test")
        assert result.total == 0
        assert result.added == 0

    def test_push_successful(self, qa_card: AnkiCard) -> None:
        from pdf2anki.anki_connect import push_cards

        with patch("pdf2anki.anki_connect._invoke") as mock_invoke:
            # ensure_deck returns deck id, addNotes returns note ids
            mock_invoke.side_effect = [123, [1001, 1002]]

            result = push_cards(
                [qa_card, qa_card], deck_name="test"
            )

        assert result.total == 2
        assert result.added == 2
        assert result.failed == 0

    def test_push_partial_failure(self, qa_card: AnkiCard) -> None:
        from pdf2anki.anki_connect import push_cards

        with patch("pdf2anki.anki_connect._invoke") as mock_invoke:
            # addNotes returns null for failed notes
            mock_invoke.side_effect = [123, [1001, None]]

            result = push_cards(
                [qa_card, qa_card], deck_name="test"
            )

        assert result.total == 2
        assert result.added == 1
        assert result.failed == 1

    def test_push_checks_anki_running(self, qa_card: AnkiCard) -> None:
        from pdf2anki.anki_connect import AnkiConnectError, push_cards

        with patch(
            "pdf2anki.anki_connect._invoke",
            side_effect=AnkiConnectError("Connection refused"),
        ):
            with pytest.raises(AnkiConnectError):
                push_cards([qa_card], deck_name="test")

    def test_push_calls_ensure_deck(self, qa_card: AnkiCard) -> None:
        from pdf2anki.anki_connect import push_cards

        with patch("pdf2anki.anki_connect._invoke") as mock_invoke:
            mock_invoke.side_effect = [123, [1001]]
            push_cards([qa_card], deck_name="my-deck")

        # First call should be createDeck
        first_call = mock_invoke.call_args_list[0]
        assert first_call[0][0] == "createDeck"
        assert first_call[1]["deck"] == "my-deck"
