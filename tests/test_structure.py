"""Tests for pdf2anki.structure - TDD RED phase.

Tests cover:
- extract_cards(): Main extraction function (with mocked Claude API)
- parse_cards_response(): Response parsing to AnkiCard list
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
    extract_cards,
    parse_cards_response,
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
# parse_cards_response Tests
# ============================================================


class TestParseCardsResponse:
    """Test response parsing from JSON to AnkiCard list."""

    def test_parse_valid_json_array(self) -> None:
        cards = parse_cards_response(SAMPLE_CARDS_JSON)
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
        cards = parse_cards_response(single)
        assert len(cards) == 1

    def test_parse_empty_array(self) -> None:
        cards = parse_cards_response("[]")
        assert cards == []

    def test_parse_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="parse"):
            parse_cards_response("not valid json{{{")

    def test_parse_non_array_raises(self) -> None:
        with pytest.raises(ValueError, match="array"):
            parse_cards_response('{"front": "q", "back": "a"}')

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
        cards = parse_cards_response(mixed)
        assert len(cards) == 1  # Only the valid card

    def test_parse_extracts_json_from_markdown(self) -> None:
        """LLM may wrap JSON in ```json blocks."""
        wrapped = '```json\n' + SAMPLE_CARDS_JSON + '\n```'
        cards = parse_cards_response(wrapped)
        assert len(cards) == 2

    def test_parse_preserves_all_fields(self) -> None:
        cards = parse_cards_response(SAMPLE_CARDS_JSON)
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
        cards = parse_cards_response(minimal)
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


# ============================================================
# extract_cards with sections (Phase 2)
# ============================================================


def _make_test_sections() -> list[Section]:
    """Create test sections for section-aware extraction tests."""
    return [
        Section(
            id="section-0",
            heading="序論",
            level=1,
            breadcrumb="正理の海 > 序論",
            text="# 序論\n\n序論の内容。因明の概要。",
            page_range="pp.1-5",
            char_count=20,
        ),
        Section(
            id="section-1",
            heading="第1章",
            level=1,
            breadcrumb="正理の海 > 第1章",
            text="# 第1章\n\n第1章の内容。論書名の意味。",
            page_range="pp.6-20",
            char_count=22,
        ),
    ]


class TestExtractCardsWithSections:
    """Test extract_cards when sections are provided (Phase 2)."""

    @patch("pdf2anki.structure._call_claude_api")
    def test_sections_each_get_api_call(self, mock_api: MagicMock) -> None:
        """Each section should generate a separate API call."""
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
        sections = _make_test_sections()
        result, tracker = extract_cards(
            text="Full document text.",
            source_file="test.pdf",
            config=config,
            sections=sections,
        )

        assert mock_api.call_count == 2  # one per section
        assert result.card_count == 2  # 1 card per section
        assert tracker.request_count == 2

    @patch("pdf2anki.structure._call_claude_api")
    def test_sections_use_build_section_prompt(
        self, mock_api: MagicMock
    ) -> None:
        """When sections are provided, build_section_prompt should be used."""
        mock_api.return_value = _make_mock_response("[]")

        config = AppConfig()
        sections = _make_test_sections()

        with patch("pdf2anki.structure.build_section_prompt") as mock_bsp:
            mock_bsp.return_value = "mocked section prompt"
            extract_cards(
                text="Full document text.",
                source_file="test.pdf",
                config=config,
                sections=sections,
            )
            assert mock_bsp.call_count == 2

    @patch("pdf2anki.structure._call_claude_api")
    def test_sections_prompt_contains_breadcrumb(
        self, mock_api: MagicMock
    ) -> None:
        """API call should include breadcrumb context from section."""
        mock_api.return_value = _make_mock_response("[]")

        config = AppConfig()
        sections = _make_test_sections()
        extract_cards(
            text="Full document text.",
            source_file="test.pdf",
            config=config,
            sections=sections,
        )

        # Check first API call's user message contains breadcrumb
        first_call = mock_api.call_args_list[0]
        messages = first_call[1]["messages"]
        user_content = messages[0]["content"]
        assert "序論" in user_content

    @patch("pdf2anki.structure._call_claude_api")
    def test_sections_model_routing_per_section(
        self, mock_api: MagicMock
    ) -> None:
        """Model should be selected based on section.char_count, not full text."""
        mock_api.return_value = _make_mock_response("[]")

        # Small sections -> should route to Haiku (< 10_000 chars)
        config = AppConfig()
        sections = _make_test_sections()
        extract_cards(
            text="A" * 50_000,  # Large full text
            source_file="test.pdf",
            config=config,
            sections=sections,
        )

        # Model in the API call should be Haiku (sections are small)
        first_call = mock_api.call_args_list[0]
        model_used = first_call[1]["model"]
        assert "haiku" in model_used

    @patch("pdf2anki.structure._call_claude_api")
    def test_sections_budget_stops_mid_processing(
        self, mock_api: MagicMock
    ) -> None:
        """Budget exceeded mid-section should stop remaining sections."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="[]")]
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.usage.input_tokens = 100_000
        mock_response.usage.output_tokens = 50_000
        mock_api.return_value = mock_response

        config = AppConfig()
        sections = _make_test_sections()
        result, tracker = extract_cards(
            text="Full text.",
            source_file="test.pdf",
            config=config,
            cost_tracker=CostTracker(budget_limit=0.001),
            sections=sections,
        )

        # Only first section processed, budget exceeded before second
        assert mock_api.call_count == 1

    @patch("pdf2anki.structure._call_claude_api")
    def test_sections_none_falls_back_to_chunks(
        self, mock_api: MagicMock
    ) -> None:
        """When sections=None, should use existing chunks path."""
        mock_api.return_value = _make_mock_response("[]")

        config = AppConfig()
        result, _ = extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
            sections=None,
        )
        assert mock_api.call_count == 1

    @patch("pdf2anki.structure._call_claude_api")
    def test_sections_empty_list_falls_back(
        self, mock_api: MagicMock
    ) -> None:
        """Empty sections list should fall back to chunks path."""
        mock_api.return_value = _make_mock_response("[]")

        config = AppConfig()
        result, _ = extract_cards(
            text="Some text.",
            source_file="test.txt",
            config=config,
            sections=[],
        )
        # Should process text directly, not iterate empty sections
        assert mock_api.call_count == 1

    @patch("pdf2anki.structure._call_claude_api")
    def test_sections_stop_at_document_card_limit(
        self, mock_api: MagicMock
    ) -> None:
        """Should stop processing sections when total cards reach config limit."""
        # Each API call returns 15 cards
        many_cards = json.dumps([
            {
                "front": f"Q{i}?",
                "back": f"A{i}.",
                "card_type": "qa",
                "bloom_level": "remember",
                "tags": ["test"],
                "related_concepts": [],
                "mnemonic_hint": None,
            }
            for i in range(15)
        ])
        mock_api.return_value = _make_mock_response(many_cards)

        # Config limit: 20 cards total
        config = AppConfig(cards_max_cards=20)
        sections = _make_test_sections()  # 2 sections
        result, _ = extract_cards(
            text="Full text.",
            source_file="test.pdf",
            config=config,
            sections=sections,
        )

        # First section: 15 cards (< 20 limit) -> continue
        # Second section: 15 + 15 = 30 would exceed, but limit check is BEFORE API call
        # So second section should NOT be called (15 >= 20 is False, but after first
        # section we have 15 cards which is < 20, so second section runs too)
        # Actually: after first section, all_cards has 15 cards, 15 < 20 so second runs
        # After second: 30 cards. For 3 sections we'd stop at 2.
        # Let's use 3 sections to properly test
        sections_3 = _make_test_sections() + [
            Section(
                id="section-2",
                heading="第2章",
                level=1,
                breadcrumb="正理の海 > 第2章",
                text="# 第2章\n\n第2章の内容。",
                page_range="pp.21-30",
                char_count=15,
            ),
        ]
        mock_api.reset_mock()
        result, _ = extract_cards(
            text="Full text.",
            source_file="test.pdf",
            config=config,
            sections=sections_3,
        )

        # After 2 sections: 30 cards >= 20 limit -> third section skipped
        assert mock_api.call_count == 2

    @patch("pdf2anki.structure._call_claude_api")
    def test_sections_inject_origin_tags(self, mock_api: MagicMock) -> None:
        """Cards from section-aware extraction should have _section:: tags."""
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
        sections = _make_test_sections()
        result, _ = extract_cards(
            text="Full text.",
            source_file="test.pdf",
            config=config,
            sections=sections,
        )

        # Each card should have _section::section-N tag injected
        for card in result.cards:
            section_tags = [t for t in card.tags if t.startswith("_section::")]
            assert len(section_tags) == 1, f"Expected _section:: tag in {card.tags}"
