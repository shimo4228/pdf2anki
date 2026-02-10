"""Vision API integration for image-aware card generation.

Builds multi-modal messages combining text and images,
calls Claude Vision API, and returns extracted cards.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic
from anthropic.types import MessageParam

from pdf2anki.config import AppConfig
from pdf2anki.cost import CostRecord, CostTracker, estimate_cost
from pdf2anki.image import ExtractedImage, image_to_base64
from pdf2anki.prompts import SYSTEM_PROMPT
from pdf2anki.schemas import ExtractionResult
from pdf2anki.structure import parse_cards_response

logger = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

## Vision / Image Analysis Instructions

When images are provided alongside the text:

1. **Analyze each image carefully**: Identify diagrams, charts, tables, \
formulas, and visual elements.
2. **Create image-aware cards**:
   - **QA cards**: Ask about what the image depicts or explains.
   - **Image occlusion cards**: Describe the image with a hidden region \
and ask what belongs there.
   - **Compare/contrast cards**: If multiple related images, compare them.
3. **Reference images**: In card front/back, describe the visual content \
textually since images cannot be embedded in Anki text fields directly.
4. **Combine text + visual**: Use surrounding text context to create \
richer, more accurate cards about the visual content.
5. **Label figures**: If the image has labels or annotations, create \
cards that test knowledge of those labels.
"""
)


def build_vision_messages(
    text: str,
    images: list[ExtractedImage],
    *,
    max_cards: int = 20,
    card_types: list[str] | None = None,
) -> list[MessageParam]:
    """Build multi-modal messages for Claude Vision API.

    Images are placed before text in the content array,
    following Claude API best practices.

    Args:
        text: Source text for card generation.
        images: Extracted images to include.
        max_cards: Maximum cards to generate.
        card_types: Specific card types to generate.

    Returns:
        List of MessageParam for Claude API.
    """
    content: list[dict[str, Any]] = []

    # Add images first (Claude processes them better this way)
    for img in images:
        if not img.image_bytes:
            continue
        b64 = image_to_base64(img)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.media_type,
                "data": b64,
            },
        })

    # Build text instruction
    parts: list[str] = []
    parts.append(
        f"Generate up to {max_cards} Anki flashcards from the "
        "following text and images."
    )

    if card_types:
        parts.append(
            f"Card types to generate: {', '.join(card_types)}"
        )

    if images:
        parts.append(
            f"The text is accompanied by {len(images)} image(s). "
            "Analyze the images alongside the text to create "
            "comprehensive flashcards."
        )

    parts.append(f"---\n\n{text.strip()}")

    content.append({"type": "text", "text": "\n\n".join(parts)})

    msg: MessageParam = {"role": "user", "content": content}  # type: ignore[typeddict-item]
    return [msg]


def extract_cards_with_vision(
    text: str,
    images: list[ExtractedImage],
    *,
    source_file: str,
    config: AppConfig,
    cost_tracker: CostTracker | None = None,
) -> tuple[ExtractionResult, CostTracker]:
    """Extract cards using Claude Vision API.

    Args:
        text: Source text.
        images: Extracted images from PDF.
        source_file: Source file name for metadata.
        config: Application configuration.
        cost_tracker: Existing tracker to accumulate costs.

    Returns:
        Tuple of (ExtractionResult, updated CostTracker).

    Raises:
        RuntimeError: If budget is exceeded.
    """
    tracker = (
        cost_tracker
        if cost_tracker is not None
        else CostTracker(budget_limit=config.cost_budget_limit)
    )

    if not tracker.is_within_budget:
        raise RuntimeError(
            f"Cost budget exceeded: ${tracker.total_cost:.4f} / "
            f"${tracker.budget_limit:.2f}"
        )

    # Use vision system prompt when images are present
    system_prompt = (
        VISION_SYSTEM_PROMPT if images else SYSTEM_PROMPT
    )

    messages = build_vision_messages(
        text=text,
        images=images,
        max_cards=config.cards_max_cards,
        card_types=config.cards_card_types,
    )

    # Vision always uses Sonnet for best image understanding
    model = config.model

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=config.max_tokens,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    )

    cards = []
    if response.content and hasattr(response.content[0], "text"):
        cards = parse_cards_response(response.content[0].text)

    cost = estimate_cost(
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    record = CostRecord(
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=cost,
    )
    tracker = tracker.add(record)

    result = ExtractionResult(
        source_file=source_file,
        cards=cards,
        model_used=model,
    )

    return result, tracker
