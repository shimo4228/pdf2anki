"""Tests for pdf2anki.vision — Vision API integration for image-aware cards."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pdf2anki.config import AppConfig
from pdf2anki.cost import CostTracker
from pdf2anki.image import ExtractedImage
from pdf2anki.vision import (
    VISION_SYSTEM_PROMPT,
    build_vision_messages,
    extract_cards_with_vision,
)


def _make_image(
    *,
    page_num: int = 0,
    index: int = 0,
    width: int = 400,
    height: int = 300,
    image_bytes: bytes = b"\x89PNG\r\n\x1a\n",
    media_type: str = "image/png",
    coverage: float = 0.3,
) -> ExtractedImage:
    return ExtractedImage(
        page_num=page_num,
        index=index,
        width=width,
        height=height,
        bbox=(0.0, 0.0, float(width), float(height)),
        image_bytes=image_bytes,
        media_type=media_type,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# VISION_SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestVisionSystemPrompt:
    """VISION_SYSTEM_PROMPT extends the base prompt."""

    def test_contains_vision_instructions(self) -> None:
        assert "image" in VISION_SYSTEM_PROMPT.lower()
        assert "visual" in VISION_SYSTEM_PROMPT.lower()

    def test_contains_base_prompt_content(self) -> None:
        assert "Wozniak" in VISION_SYSTEM_PROMPT
        assert "Anki" in VISION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# build_vision_messages
# ---------------------------------------------------------------------------


class TestBuildVisionMessages:
    """build_vision_messages constructs multi-modal Claude messages."""

    def test_text_only(self) -> None:
        messages = build_vision_messages(
            text="Some text about deep learning.",
            images=[],
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        # Only text block when no images
        text_blocks = [b for b in content if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert "deep learning" in text_blocks[0]["text"]

    def test_single_image(self) -> None:
        img = _make_image()
        messages = build_vision_messages(
            text="Explain this diagram.",
            images=[img],
        )
        content = messages[0]["content"]
        image_blocks = [b for b in content if b["type"] == "image"]
        text_blocks = [b for b in content if b["type"] == "text"]
        assert len(image_blocks) == 1
        assert image_blocks[0]["source"]["type"] == "base64"
        assert image_blocks[0]["source"]["media_type"] == "image/png"
        assert len(text_blocks) == 1

    def test_multiple_images(self) -> None:
        images = [_make_image(index=i) for i in range(3)]
        messages = build_vision_messages(
            text="Analyze these figures.",
            images=images,
        )
        content = messages[0]["content"]
        image_blocks = [b for b in content if b["type"] == "image"]
        assert len(image_blocks) == 3

    def test_max_cards_in_prompt(self) -> None:
        messages = build_vision_messages(
            text="Some text.",
            images=[_make_image()],
            max_cards=10,
        )
        content = messages[0]["content"]
        text_blocks = [b for b in content if b["type"] == "text"]
        assert "10" in text_blocks[0]["text"]

    def test_images_before_text(self) -> None:
        """Images should come before text in content array."""
        img = _make_image()
        messages = build_vision_messages(
            text="Explain.",
            images=[img],
        )
        content = messages[0]["content"]
        # First block(s) should be images, last should be text
        assert content[0]["type"] == "image"
        assert content[-1]["type"] == "text"

    def test_empty_image_bytes_skipped(self) -> None:
        img = _make_image(image_bytes=b"")
        messages = build_vision_messages(
            text="Some text.",
            images=[img],
        )
        content = messages[0]["content"]
        image_blocks = [b for b in content if b["type"] == "image"]
        assert len(image_blocks) == 0


# ---------------------------------------------------------------------------
# extract_cards_with_vision
# ---------------------------------------------------------------------------


class TestExtractCardsWithVision:
    """extract_cards_with_vision calls Claude Vision API and parses cards."""

    @patch("pdf2anki.vision.anthropic.Anthropic")
    def test_returns_extraction_result(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 200
        mock_response.content = [
            MagicMock(
                text='[{"front": "What is ReLU?", "back": "max(0, x)", '
                '"card_type": "qa", "bloom_level": "remember", '
                '"tags": ["AI::activation"], "related_concepts": [], '
                '"mnemonic_hint": null, "media": []}]'
            )
        ]
        mock_client.messages.create.return_value = mock_response

        config = AppConfig()
        tracker = CostTracker(budget_limit=1.0)
        img = _make_image()

        result, updated_tracker = extract_cards_with_vision(
            text="ReLU activation function",
            images=[img],
            source_file="test.pdf",
            config=config,
            cost_tracker=tracker,
        )

        assert result.card_count == 1
        assert result.cards[0].front == "What is ReLU?"
        assert updated_tracker.request_count == 1

    @patch("pdf2anki.vision.anthropic.Anthropic")
    def test_no_images_falls_through(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """With no images, should still work (text-only vision call)."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.content = [MagicMock(text="[]")]
        mock_client.messages.create.return_value = mock_response

        config = AppConfig()
        tracker = CostTracker(budget_limit=1.0)

        result, _ = extract_cards_with_vision(
            text="Some text.",
            images=[],
            source_file="test.pdf",
            config=config,
            cost_tracker=tracker,
        )

        assert result.card_count == 0

    @patch("pdf2anki.vision.anthropic.Anthropic")
    def test_vision_system_prompt_used(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.content = [MagicMock(text="[]")]
        mock_client.messages.create.return_value = mock_response

        config = AppConfig()
        tracker = CostTracker(budget_limit=1.0)

        extract_cards_with_vision(
            text="Text.",
            images=[_make_image()],
            source_file="test.pdf",
            config=config,
            cost_tracker=tracker,
        )

        call_kwargs = mock_client.messages.create.call_args
        system_arg = call_kwargs.kwargs.get("system", [])
        system_text = system_arg[0]["text"] if system_arg else ""
        assert "image" in system_text.lower() or "visual" in system_text.lower()

    def test_budget_exceeded_raises(self) -> None:
        config = AppConfig(cost_budget_limit=0.01)
        # Tracker already over budget
        from pdf2anki.cost import CostRecord

        record = CostRecord(
            model="test",
            input_tokens=1000,
            output_tokens=1000,
            cost_usd=0.02,
        )
        tracker = CostTracker(budget_limit=0.01, records=(record,))

        with pytest.raises(RuntimeError, match="budget"):
            extract_cards_with_vision(
                text="Text.",
                images=[],
                source_file="test.pdf",
                config=config,
                cost_tracker=tracker,
            )
