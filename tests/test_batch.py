"""Tests for pdf2anki.batch - TDD RED phase.

Tests cover:
- BatchRequest / BatchResult frozen dataclasses
- create_batch_requests(): builds API-ready batch from sections
- submit_batch(): submits batch to Anthropic API (mocked)
- poll_batch(): polls until processing completes (mocked)
- collect_batch_results(): retrieves and parses results (mocked)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pdf2anki.batch import (
    BatchRequest,
    BatchResult,
    collect_batch_results,
    create_batch_requests,
    poll_batch,
    submit_batch,
)
from pdf2anki.config import AppConfig
from pdf2anki.schemas import AnkiCard, CardType
from pdf2anki.section import Section

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
        "related_concepts": ["deep learning"],
        "mnemonic_hint": None,
    },
])


def _make_test_sections() -> list[Section]:
    """Create test sections for batch tests."""
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


# ============================================================
# BatchRequest Tests
# ============================================================


class TestBatchRequest:
    """Test BatchRequest frozen dataclass."""

    def test_create_batch_request(self) -> None:
        req = BatchRequest(
            custom_id="section-0",
            model="claude-haiku-4-5-20251001",
            user_prompt="Generate cards from...",
            system_prompt="You are an expert...",
            max_tokens=8192,
        )
        assert req.custom_id == "section-0"
        assert req.model == "claude-haiku-4-5-20251001"
        assert req.max_tokens == 8192

    def test_frozen_immutability(self) -> None:
        req = BatchRequest(
            custom_id="section-0",
            model="claude-haiku-4-5-20251001",
            user_prompt="test",
            system_prompt="test",
            max_tokens=8192,
        )
        with pytest.raises(AttributeError):
            req.custom_id = "other"  # type: ignore[misc]


# ============================================================
# BatchResult Tests
# ============================================================


class TestBatchResult:
    """Test BatchResult frozen dataclass."""

    def test_create_batch_result(self) -> None:
        result = BatchResult(
            custom_id="section-0",
            cards=[],
            input_tokens=500,
            output_tokens=300,
            model="claude-haiku-4-5-20251001",
        )
        assert result.custom_id == "section-0"
        assert result.cards == []
        assert result.input_tokens == 500

    def test_frozen_immutability(self) -> None:
        result = BatchResult(
            custom_id="section-0",
            cards=[],
            input_tokens=0,
            output_tokens=0,
            model="test",
        )
        with pytest.raises(AttributeError):
            result.custom_id = "other"  # type: ignore[misc]

    def test_result_with_cards(self) -> None:
        card = AnkiCard(
            front="Q?",
            back="A.",
            card_type=CardType.QA,
            bloom_level="remember",
            tags=["test"],
        )
        result = BatchResult(
            custom_id="section-0",
            cards=[card],
            input_tokens=100,
            output_tokens=50,
            model="test",
        )
        assert len(result.cards) == 1
        assert result.cards[0].front == "Q?"


# ============================================================
# create_batch_requests Tests
# ============================================================


class TestCreateBatchRequests:
    """Test create_batch_requests() function."""

    def test_creates_one_request_per_section(self) -> None:
        sections = _make_test_sections()
        config = AppConfig()
        requests = create_batch_requests(
            sections, document_title="正理の海", config=config
        )
        assert len(requests) == 2

    def test_custom_id_matches_section_id(self) -> None:
        sections = _make_test_sections()
        config = AppConfig()
        requests = create_batch_requests(
            sections, document_title="正理の海", config=config
        )
        assert requests[0].custom_id == "section-0"
        assert requests[1].custom_id == "section-1"

    def test_user_prompt_contains_breadcrumb(self) -> None:
        sections = _make_test_sections()
        config = AppConfig()
        requests = create_batch_requests(
            sections, document_title="正理の海", config=config
        )
        assert "序論" in requests[0].user_prompt
        assert "第1章" in requests[1].user_prompt

    def test_model_routing_per_section(self) -> None:
        """Small sections route to Haiku, large sections to Sonnet."""
        small_section = Section(
            id="section-0",
            heading="Small",
            level=1,
            breadcrumb="Doc > Small",
            text="Short content.",
            page_range="",
            char_count=100,
        )
        large_section = Section(
            id="section-1",
            heading="Large",
            level=1,
            breadcrumb="Doc > Large",
            text="X" * 15_000,
            page_range="",
            char_count=15_000,
        )
        config = AppConfig()
        requests = create_batch_requests(
            [small_section, large_section],
            document_title="Doc",
            config=config,
        )
        assert "haiku" in requests[0].model
        assert "sonnet" in requests[1].model

    def test_force_model_overrides_routing(self) -> None:
        """When model_overridden=True, all sections use the forced model."""
        sections = _make_test_sections()
        config = AppConfig(model="claude-opus-4-6", model_overridden=True)
        requests = create_batch_requests(
            sections, document_title="正理の海", config=config
        )
        for req in requests:
            assert req.model == "claude-opus-4-6"

    def test_empty_sections_returns_empty(self) -> None:
        config = AppConfig()
        requests = create_batch_requests(
            [], document_title="Doc", config=config
        )
        assert requests == []

    def test_max_tokens_from_config(self) -> None:
        sections = _make_test_sections()
        config = AppConfig(max_tokens=4096)
        requests = create_batch_requests(
            sections, document_title="Doc", config=config
        )
        for req in requests:
            assert req.max_tokens == 4096

    def test_focus_topics_forwarded(self) -> None:
        sections = _make_test_sections()
        config = AppConfig()
        requests = create_batch_requests(
            sections,
            document_title="Doc",
            config=config,
            focus_topics=["因明", "論理学"],
        )
        assert "因明" in requests[0].user_prompt

    def test_system_prompt_is_system_prompt(self) -> None:
        """All requests should include the standard SYSTEM_PROMPT."""
        from pdf2anki.prompts import SYSTEM_PROMPT

        sections = _make_test_sections()
        config = AppConfig()
        requests = create_batch_requests(
            sections, document_title="Doc", config=config
        )
        for req in requests:
            assert req.system_prompt == SYSTEM_PROMPT


# ============================================================
# submit_batch Tests
# ============================================================


class TestSubmitBatch:
    """Test submit_batch() with mocked Anthropic client."""

    def test_returns_batch_id(self) -> None:
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.id = "msgbatch_test123"
        mock_client.messages.batches.create.return_value = mock_batch

        req = BatchRequest(
            custom_id="section-0",
            model="claude-haiku-4-5-20251001",
            user_prompt="test prompt",
            system_prompt="system prompt",
            max_tokens=8192,
        )
        batch_id = submit_batch([req], client=mock_client)
        assert batch_id == "msgbatch_test123"

    def test_sends_correct_request_format(self) -> None:
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.id = "msgbatch_abc"
        mock_client.messages.batches.create.return_value = mock_batch

        req = BatchRequest(
            custom_id="section-0",
            model="claude-haiku-4-5-20251001",
            user_prompt="Generate cards...",
            system_prompt="You are an expert...",
            max_tokens=4096,
        )
        submit_batch([req], client=mock_client)

        call_kwargs = mock_client.messages.batches.create.call_args[1]
        requests = call_kwargs["requests"]
        assert len(requests) == 1
        assert requests[0]["custom_id"] == "section-0"
        assert requests[0]["params"]["model"] == "claude-haiku-4-5-20251001"
        assert requests[0]["params"]["max_tokens"] == 4096
        assert requests[0]["params"]["messages"][0]["role"] == "user"

    def test_system_prompt_with_cache_control(self) -> None:
        """System prompt should include cache_control for prompt caching."""
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.id = "msgbatch_abc"
        mock_client.messages.batches.create.return_value = mock_batch

        req = BatchRequest(
            custom_id="section-0",
            model="claude-haiku-4-5-20251001",
            user_prompt="test",
            system_prompt="You are an expert...",
            max_tokens=8192,
        )
        submit_batch([req], client=mock_client)

        call_kwargs = mock_client.messages.batches.create.call_args[1]
        system = call_kwargs["requests"][0]["params"]["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_multiple_requests_submitted(self) -> None:
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.id = "msgbatch_multi"
        mock_client.messages.batches.create.return_value = mock_batch

        reqs = [
            BatchRequest(
                custom_id=f"section-{i}",
                model="claude-haiku-4-5-20251001",
                user_prompt=f"prompt {i}",
                system_prompt="system",
                max_tokens=8192,
            )
            for i in range(5)
        ]
        batch_id = submit_batch(reqs, client=mock_client)
        assert batch_id == "msgbatch_multi"

        call_kwargs = mock_client.messages.batches.create.call_args[1]
        assert len(call_kwargs["requests"]) == 5

    def test_empty_requests_raises(self) -> None:
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="empty"):
            submit_batch([], client=mock_client)


# ============================================================
# poll_batch Tests
# ============================================================


class TestPollBatch:
    """Test poll_batch() with mocked Anthropic client."""

    def test_returns_immediately_when_ended(self) -> None:
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.processing_status = "ended"
        mock_client.messages.batches.retrieve.return_value = mock_batch

        result = poll_batch(
            "msgbatch_123",
            client=mock_client,
            poll_interval=0.01,
            timeout=1.0,
        )
        assert result.processing_status == "ended"

    @patch("pdf2anki.batch.time.sleep")
    def test_polls_until_ended(self, mock_sleep: MagicMock) -> None:
        mock_client = MagicMock()

        in_progress = MagicMock()
        in_progress.processing_status = "in_progress"
        ended = MagicMock()
        ended.processing_status = "ended"

        mock_client.messages.batches.retrieve.side_effect = [
            in_progress,
            in_progress,
            ended,
        ]

        result = poll_batch(
            "msgbatch_123",
            client=mock_client,
            poll_interval=0.01,
            timeout=10.0,
        )
        assert result.processing_status == "ended"
        assert mock_client.messages.batches.retrieve.call_count == 3

    @patch("pdf2anki.batch.time.sleep")
    def test_timeout_raises(self, mock_sleep: MagicMock) -> None:
        """Should raise TimeoutError when batch doesn't complete in time."""
        mock_client = MagicMock()
        in_progress = MagicMock()
        in_progress.processing_status = "in_progress"
        mock_client.messages.batches.retrieve.return_value = in_progress

        # Simulate time passing beyond timeout
        with patch("pdf2anki.batch.time.monotonic") as mock_time:
            mock_time.side_effect = [0.0, 0.5, 1.5, 2.5]  # exceed 2.0 timeout
            with pytest.raises(TimeoutError, match="Batch"):
                poll_batch(
                    "msgbatch_123",
                    client=mock_client,
                    poll_interval=0.01,
                    timeout=2.0,
                )


# ============================================================
# collect_batch_results Tests
# ============================================================


class TestCollectBatchResults:
    """Test collect_batch_results() with mocked Anthropic client."""

    def _make_result_entry(
        self,
        custom_id: str,
        content_text: str,
        *,
        succeeded: bool = True,
        input_tokens: int = 500,
        output_tokens: int = 300,
        model: str = "claude-haiku-4-5-20251001",
    ) -> MagicMock:
        entry = MagicMock()
        entry.custom_id = custom_id
        if succeeded:
            entry.result.type = "succeeded"
            content_block = MagicMock()
            content_block.text = content_text
            entry.result.message.content = [content_block]
            entry.result.message.model = model
            entry.result.message.usage.input_tokens = input_tokens
            entry.result.message.usage.output_tokens = output_tokens
        else:
            entry.result.type = "errored"
        return entry

    def test_parses_successful_results(self) -> None:
        mock_client = MagicMock()

        entry = self._make_result_entry("section-0", SAMPLE_CARDS_JSON)
        mock_client.messages.batches.results.return_value = [entry]

        results = collect_batch_results("msgbatch_123", client=mock_client)
        assert len(results) == 1
        assert results[0].custom_id == "section-0"
        assert len(results[0].cards) == 1
        assert results[0].cards[0].front == "What is a neural network?"

    def test_skips_errored_results(self) -> None:
        mock_client = MagicMock()

        ok_entry = self._make_result_entry("section-0", SAMPLE_CARDS_JSON)
        err_entry = self._make_result_entry(
            "section-1", "", succeeded=False
        )
        mock_client.messages.batches.results.return_value = [
            ok_entry, err_entry
        ]

        results = collect_batch_results("msgbatch_123", client=mock_client)
        assert len(results) == 1
        assert results[0].custom_id == "section-0"

    def test_records_token_usage(self) -> None:
        mock_client = MagicMock()

        entry = self._make_result_entry(
            "section-0",
            SAMPLE_CARDS_JSON,
            input_tokens=1000,
            output_tokens=500,
        )
        mock_client.messages.batches.results.return_value = [entry]

        results = collect_batch_results("msgbatch_123", client=mock_client)
        assert results[0].input_tokens == 1000
        assert results[0].output_tokens == 500

    def test_records_model(self) -> None:
        mock_client = MagicMock()

        entry = self._make_result_entry(
            "section-0",
            SAMPLE_CARDS_JSON,
            model="claude-sonnet-4-5-20250929",
        )
        mock_client.messages.batches.results.return_value = [entry]

        results = collect_batch_results("msgbatch_123", client=mock_client)
        assert results[0].model == "claude-sonnet-4-5-20250929"

    def test_empty_results(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.batches.results.return_value = []

        results = collect_batch_results("msgbatch_123", client=mock_client)
        assert results == []

    def test_invalid_json_in_result_skipped(self) -> None:
        """Results with unparseable JSON should be skipped, not crash."""
        mock_client = MagicMock()

        bad_entry = self._make_result_entry("section-0", "not valid json{{{")
        good_entry = self._make_result_entry("section-1", SAMPLE_CARDS_JSON)
        mock_client.messages.batches.results.return_value = [
            bad_entry, good_entry
        ]

        results = collect_batch_results("msgbatch_123", client=mock_client)
        assert len(results) == 1
        assert results[0].custom_id == "section-1"

    def test_multiple_successful_results(self) -> None:
        mock_client = MagicMock()

        entries = [
            self._make_result_entry(f"section-{i}", SAMPLE_CARDS_JSON)
            for i in range(3)
        ]
        mock_client.messages.batches.results.return_value = entries

        results = collect_batch_results("msgbatch_123", client=mock_client)
        assert len(results) == 3
        total_cards = sum(len(r.cards) for r in results)
        assert total_cards == 3  # 1 card per result
