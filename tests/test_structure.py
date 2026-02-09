"""Tests for pdf2anki.structure - TDD RED phase.

Tests cover:
- extract_cards(): Main extraction function (with mocked Claude API)
- _parse_cards_response(): Response parsing to AnkiCard list
- Retry logic on API errors
- Budget enforcement (stop if budget exceeded)
- Multi-chunk processing
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from pydantic import ValidationError

from pdf2anki.config import AppConfig
from pdf2anki.cost import CostRecord, CostTracker
from pdf2anki.schemas import AnkiCard, BloomLevel, CardType, ExtractionResult
from pdf2anki.section import Section
from pdf2anki.structure import (
    _parse_cards_response,
    extract_cards,
)

# ============================================================
# Fixtures
# ============================================================

SAMPLE_CARDS_JSON = json.dumps([
    {
        "front": "What is a neural network?",
        "back": "A computational model inspired by biological neural networks.",
        "card_type": "qa",
        "bloom_level": "understand",
        "tags": ["AI::basics"],
        "related_concepts": ["deep learning", "perceptron"],
        "mnemonic_hint": None,
    },
    {
        "front": (
            "{{c1::Gradient descent}} is an optimization"
            " algorithm that minimizes the loss function."
        ),
        "back": "",
        "card_type": "cloze",
        "bloom_level": "remember",
        "tags": ["AI::optimization"],
        "related_concepts": ["learning rate", "SGD"],
        "mnemonic_hint": "Gradient = slope, Descent = going down",
    },
])


def _make_mock_response(
    content_text: str,
    input_tokens: int = 500,
    output_tokens: int = 300,
) -> MagicMock:
    """Create a mock Anthropic API response."""
    mock_response = MagicMock()
    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = content_text
    mock_response.content = [mock_content_block]
    mock_response.model = "claude-sonnet-4-5-20250929"
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    return mock_response


# ============================================================
# _parse_cards_response Tests
# ============================================================


class TestParseCardsResponse:
    """Test response parsing from JSON to AnkiCard list."""

    def test_parse_valid_json_array(self) -> None:
        cards = _parse_cards_response(SAMPLE_CARDS_JSON)
        assert len(cards) == 2
        assert isinstance(cards[0], AnkiCard)
        assert cards[0].card_type == CardType.QA
        assert cards[1].card_type == CardType.CLOZE

    def test_parse_single_card(self) -> None:
        single = json.dumps([{
            "front": "What is ML?",
            "back": "Machine Learning.",
            "card_type": "qa",
            "bloom_level": "remember",
            "tags": ["ML"],
            "related_concepts": [],
            "mnemonic_hint": None,
        }])
        cards = _parse_cards_response(single)
        assert len(cards) == 1

    def test_parse_empty_array(self) -> None:
        cards = _parse_cards_response("[]")
        assert cards == []

    def test_parse_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="parse"):
            _parse_cards_response("not valid json{{{")

    def test_parse_non_array_raises(self) -> None:
        with pytest.raises(ValueError, match="array"):
            _parse_cards_response('{"front": "q", "back": "a"}')

    def test_parse_skips_invalid_cards(self) -> None:
        """Cards with validation errors should be skipped, not crash."""
        mixed = json.dumps([
            {
                "front": "Valid card?",
                "back": "Yes.",
                "card_type": "qa",
                "bloom_level": "remember",
                "tags": ["test"],
                "related_concepts": [],
                "mnemonic_hint": None,
            },
            {
                "front": "",  # Invalid: empty front
                "back": "answer",
                "card_type": "qa",
                "bloom_level": "remember",
                "tags": ["test"],
                "related_concepts": [],
                "mnemonic_hint": None,
            },
        ])
        cards = _parse_cards_response(mixed)
        assert len(cards) == 1  # Only the valid card

    def test_parse_extracts_json_from_markdown(self) -> None:
        """LLM may wrap JSON in ```json blocks."""
        wrapped = '```json\n' + SAMPLE_CARDS_JSON + '\n```'
        cards = _parse_cards_response(wrapped)
        assert len(cards) == 2

    def test_parse_preserves_all_fields(self) -> None:
        cards = _parse_cards_response(SAMPLE_CARDS_JSON)
        card = cards[0]
        assert card.front == "What is a neural network?"
        assert card.bloom_level == BloomLevel.UNDERSTAND
        assert "AI::basics" in card.tags
        assert "deep learning" in card.related_concepts

    def test_parse_handles_missing_optional_fields(self) -> None:
        """Cards missing optional fields should get defaults."""
        minimal = json.dumps([{
            "front": "Q?",
            "back": "A.",
            "card_type": "qa",
            "bloom_level": "remember",
            "tags": ["t"],
        }])
        cards = _parse_cards_response(minimal)
        assert len(cards) == 1
        assert cards[0].related_concepts == []
        assert cards[0].mnemonic_hint is None


# ============================================================
# extract_cards Tests (with mocked API)
# ============================================================


class TestExtractCards:
    """Test extract_cards with mocked Claude API."""

    @patch("pdf2anki.structure._call_claude_api")
    def test_basic_extraction(self, mock_api: MagicMock) -> None:
        mock_api.return_value = _make_mock_response(SAMPLE_CARDS_JSON)

        config = AppConfig()
        result, tracker = extract_cards(
            text="Neural networks are computational models...",
            source_file="test.txt",
            config=config,
        )

        assert isinstance(result, ExtractionResult)
        assert result.card_count == 2
        assert result.source_file == "test.txt"
        assert isinstance(tracker, CostTracker)
        assert tracker.request_count == 1
        assert tracker.total_cost > 0

    @patch("pdf2anki.structure._call_claude_api")
    def test_returns_immutable_result(self, mock_api: MagicMock) -> None:
        mock_api.return_value = _make_mock_response(SAMPLE_CARDS_JSON)
        config = AppConfig()
        result, _ = extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
        )
        assert isinstance(result, ExtractionResult)
        # ExtractionResult is frozen
        with pytest.raises(ValidationError):
            result.source_file = "other"  # type: ignore[misc]

    @patch("pdf2anki.structure._call_claude_api")
    def test_existing_tracker_preserved(self, mock_api: MagicMock) -> None:
        """Passing an existing tracker should accumulate costs."""
        mock_api.return_value = _make_mock_response(SAMPLE_CARDS_JSON)

        existing_record = CostRecord(
            model="test", input_tokens=100, output_tokens=50, cost_usd=0.01
        )
        existing_tracker = CostTracker(budget_limit=1.00).add(existing_record)

        config = AppConfig()
        _, tracker = extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
            cost_tracker=existing_tracker,
        )
        assert tracker.request_count == 2  # existing + new
        assert tracker.total_cost > 0.01

    @patch("pdf2anki.structure._call_claude_api")
    def test_budget_exceeded_raises(self, mock_api: MagicMock) -> None:
        """Should raise when budget is already exceeded."""
        big_record = CostRecord(
            model="test", input_tokens=0, output_tokens=0, cost_usd=100.0
        )
        exhausted_tracker = CostTracker(budget_limit=1.00).add(big_record)

        config = AppConfig()
        with pytest.raises(RuntimeError, match="budget"):
            extract_cards(
                text="Some text.",
                source_file="test.txt",
                config=config,
                cost_tracker=exhausted_tracker,
            )

    @patch("pdf2anki.structure._call_claude_api")
    def test_api_error_retries(self, mock_api: MagicMock) -> None:
        """Should retry on API errors up to max retries."""
        mock_api.side_effect = [
            anthropic.APIConnectionError(request=MagicMock()),
            _make_mock_response(SAMPLE_CARDS_JSON),
        ]

        config = AppConfig()
        result, _ = extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
        )
        assert result.card_count == 2
        assert mock_api.call_count == 2

    @patch("pdf2anki.structure._call_claude_api")
    def test_api_error_exhausts_retries(self, mock_api: MagicMock) -> None:
        """Should raise after all retries are exhausted."""
        mock_api.side_effect = anthropic.APIConnectionError(
            request=MagicMock()
        )

        config = AppConfig()
        with pytest.raises(RuntimeError, match="API"):
            extract_cards(
                text="Some text.",
                source_file="test.txt",
                config=config,
            )

    @patch("pdf2anki.structure._call_claude_api")
    def test_empty_response_returns_no_cards(self, mock_api: MagicMock) -> None:
        mock_api.return_value = _make_mock_response("[]")

        config = AppConfig()
        result, _ = extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
        )
        assert result.card_count == 0

    @patch("pdf2anki.structure._call_claude_api")
    def test_model_recorded_in_result(self, mock_api: MagicMock) -> None:
        mock_api.return_value = _make_mock_response(SAMPLE_CARDS_JSON)

        config = AppConfig()
        result, _ = extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
        )
        assert "claude" in result.model_used

    @patch("pdf2anki.structure._call_claude_api")
    def test_focus_topics_forwarded(self, mock_api: MagicMock) -> None:
        """focus_topics should be included in the API call."""
        mock_api.return_value = _make_mock_response("[]")

        config = AppConfig()
        extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
            focus_topics=["CNN", "RNN"],
        )
        # Verify the user prompt contains focus topics
        call_kwargs = mock_api.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][3]
        user_content = messages[0]["content"]
        assert "CNN" in user_content
        assert "RNN" in user_content

    @patch("pdf2anki.structure._call_claude_api")
    def test_multi_chunk_processing(self, mock_api: MagicMock) -> None:
        """Multiple chunks should each generate a separate API call."""
        mock_api.return_value = _make_mock_response(
            json.dumps([{
                "front": "Q?",
                "back": "A.",
                "card_type": "qa",
                "bloom_level": "remember",
                "tags": ["test"],
                "related_concepts": [],
                "mnemonic_hint": None,
            }])
        )

        config = AppConfig()
        result, tracker = extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
            chunks=["Chunk 1 text.", "Chunk 2 text."],
        )
        assert mock_api.call_count == 2
        assert result.card_count == 2  # 1 per chunk
        assert tracker.request_count == 2

    @patch("pdf2anki.structure._call_claude_api")
    def test_empty_content_response_skipped(self, mock_api: MagicMock) -> None:
        """Empty content should be skipped without error."""
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 0
        mock_api.return_value = mock_response

        config = AppConfig()
        result, _ = extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
        )
        assert result.card_count == 0

    @patch("pdf2anki.structure._call_claude_api")
    def test_max_cards_respected(self, mock_api: MagicMock) -> None:
        """Config max_cards should be passed to the prompt."""
        mock_api.return_value = _make_mock_response("[]")

        config = AppConfig(cards_max_cards=10)
        extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
        )
        call_kwargs = mock_api.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][3]
        user_content = messages[0]["content"]
        assert "10" in user_content

    @patch("pdf2anki.structure._call_claude_api")
    def test_budget_exceeded_stops_mid_processing(self, mock_api: MagicMock) -> None:
        """Should stop processing remaining chunks when budget runs out mid-loop."""
        from pdf2anki.cost import CostTracker

        # Return a response that costs money, pushing over the tiny budget
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="[]")]
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.usage.input_tokens = 100_000
        mock_response.usage.output_tokens = 50_000
        mock_api.return_value = mock_response

        # Budget of $0.001 - first chunk will exceed it
        config = AppConfig()
        result, tracker = extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
            cost_tracker=CostTracker(budget_limit=0.001),
            chunks=["chunk1", "chunk2"],
        )
        # Only first chunk processed, second stopped by budget check
        assert mock_api.call_count == 1

    @patch("pdf2anki.structure._call_claude_api")
    def test_cost_warn_at_threshold_logged(
        self, mock_api: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log warning when cost approaches budget limit."""
        from pdf2anki.cost import CostRecord, CostTracker

        mock_api.return_value = _make_mock_response(SAMPLE_CARDS_JSON)

        # Create tracker at 85% of budget (above default 80% warn threshold)
        near_limit_tracker = CostTracker(budget_limit=1.0).add(
            CostRecord(model="test", input_tokens=0, output_tokens=0, cost_usd=0.85)
        )

        config = AppConfig()
        with caplog.at_level("WARNING", logger="pdf2anki.structure"):
            extract_cards(
                text="Some text.",
                source_file="test.txt",
                config=config,
                cost_tracker=near_limit_tracker,
            )
        assert any("Cost approaching budget limit" in msg for msg in caplog.messages)

    @patch("pdf2anki.structure._call_claude_api")
    def test_non_text_block_response_skipped(self, mock_api: MagicMock) -> None:
        """Response with non-text block should be skipped."""
        mock_response = MagicMock()
        # Block without 'text' attribute (e.g., ToolUseBlock)
        mock_block = MagicMock(spec=[])  # empty spec = no attributes
        mock_response.content = [mock_block]
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_api.return_value = mock_response

        config = AppConfig()
        result, _ = extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
        )
        assert result.card_count == 0
