"""LLM-based structured card extraction for pdf2anki.

Calls Claude API to generate Anki cards from text using Structured Outputs.
Includes retry logic, budget enforcement, prompt caching, and multi-chunk
processing.
"""

from __future__ import annotations

import json
import logging
import re
import time

import anthropic
from anthropic.types import MessageParam
from pydantic import ValidationError

from pdf2anki.config import DEFAULT_MODEL, AppConfig
from pdf2anki.cost import CostRecord, CostTracker, estimate_cost, select_model
from pdf2anki.prompts import SYSTEM_PROMPT, build_section_prompt, build_user_prompt
from pdf2anki.schemas import AnkiCard, ExtractionResult
from pdf2anki.section import Section

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2.0

# Retryable API errors (network/rate-limit/server errors only)
_RETRYABLE_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

# Regex to extract JSON from markdown code blocks
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def parse_cards_response(response_text: str) -> list[AnkiCard]:
    """Parse LLM response text into a list of AnkiCard objects.

    Handles:
    - Direct JSON arrays
    - JSON wrapped in ```json code blocks
    - Skips invalid cards (logs warning instead of crashing)

    Args:
        response_text: Raw text response from the LLM.

    Returns:
        List of validated AnkiCard objects.

    Raises:
        ValueError: If response is not valid JSON or not an array.
    """
    text = response_text.strip()

    # Extract JSON from markdown code blocks if present
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array of cards, got {type(data).__name__}"
        )

    cards: list[AnkiCard] = []
    for i, item in enumerate(data):
        try:
            card = AnkiCard.model_validate(item)
            cards.append(card)
        except (ValidationError, TypeError) as e:
            logger.warning("Skipping invalid card at index %d: %s", i, e)

    return cards


def _call_claude_api(
    *,
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    messages: list[MessageParam],
) -> anthropic.types.Message:
    """Call the Claude API with prompt caching for the system prompt.

    Args:
        client: Anthropic client instance.
        model: Model ID to use.
        max_tokens: Maximum output tokens.
        messages: User messages.

    Returns:
        Claude API Message response.
    """
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    )


def extract_cards(
    text: str,
    *,
    source_file: str,
    config: AppConfig,
    cost_tracker: CostTracker | None = None,
    chunks: list[str] | None = None,
    sections: list[Section] | None = None,
    focus_topics: list[str] | None = None,
    bloom_filter: list[str] | None = None,
    additional_tags: list[str] | None = None,
) -> tuple[ExtractionResult, CostTracker]:
    """Extract Anki cards from text using Claude API.

    When sections are provided (non-empty), uses section-aware prompts
    with breadcrumb context and per-section model routing. Otherwise,
    falls back to chunk-based processing.

    Args:
        text: Source text (used when chunks/sections are None).
        source_file: Source file name for metadata.
        config: Application configuration.
        cost_tracker: Existing tracker to accumulate costs.
        chunks: Pre-split text chunks (overrides text if provided).
        sections: Section objects with structural metadata.
        focus_topics: Topics to emphasize in generation.
        bloom_filter: Only generate cards at these Bloom levels.
        additional_tags: Extra tags for all cards.

    Returns:
        Tuple of (ExtractionResult, updated CostTracker).

    Raises:
        RuntimeError: If budget is exceeded or API calls fail after retries.
    """
    tracker = cost_tracker if cost_tracker is not None else CostTracker(
        budget_limit=config.cost_budget_limit
    )

    if not tracker.is_within_budget:
        raise RuntimeError(
            f"Cost budget exceeded: ${tracker.total_cost:.4f} / "
            f"${tracker.budget_limit:.2f}"
        )

    # Section-aware path: per-section model routing + breadcrumb prompts
    if sections:
        return _extract_cards_from_sections(
            sections=sections,
            source_file=source_file,
            document_title=source_file,
            config=config,
            tracker=tracker,
            focus_topics=focus_topics,
            bloom_filter=bloom_filter,
            additional_tags=additional_tags,
        )

    # Legacy chunk-based path
    model = select_model(
        text_length=len(text),
        card_count=config.cards_max_cards,
        force_model=config.model if config.model_overridden else None,
    )

    text_chunks = chunks if chunks is not None else [text]
    warn_threshold = config.cost_warn_at * config.cost_budget_limit
    cost_warned = False

    client = anthropic.Anthropic()
    all_cards: list[AnkiCard] = []

    for chunk in text_chunks:
        if not tracker.is_within_budget:
            logger.warning("Budget exceeded, stopping chunk processing")
            break

        if not cost_warned and tracker.total_cost >= warn_threshold:
            logger.warning(
                "Cost approaching budget limit: $%.4f / $%.2f (%.0f%%)",
                tracker.total_cost,
                tracker.budget_limit,
                (tracker.total_cost / tracker.budget_limit) * 100,
            )
            cost_warned = True

        user_prompt = build_user_prompt(
            chunk,
            max_cards=config.cards_max_cards,
            card_types=config.cards_card_types,
            focus_topics=focus_topics,
            bloom_filter=bloom_filter,
            additional_tags=additional_tags,
        )

        messages: list[MessageParam] = [
            {"role": "user", "content": user_prompt}
        ]

        response = _call_with_retry(
            client=client,
            model=model,
            max_tokens=config.max_tokens,
            messages=messages,
        )

        if not response.content:
            logger.warning("Empty API response for chunk, skipping")
            continue

        first_block = response.content[0]
        if not hasattr(first_block, "text"):
            logger.warning(
                "Unexpected response block type: %s",
                type(first_block).__name__,
            )
            continue

        response_text = first_block.text
        cards = parse_cards_response(response_text)
        all_cards.extend(cards)

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
        cards=all_cards,
        model_used=model,
    )

    return result, tracker


_SECTION_MAX_CARDS = 20  # Section-level cap (vs document-level 50)


def _extract_cards_from_sections(
    *,
    sections: list[Section],
    source_file: str,
    document_title: str,
    config: AppConfig,
    tracker: CostTracker,
    focus_topics: list[str] | None,
    bloom_filter: list[str] | None,
    additional_tags: list[str] | None,
) -> tuple[ExtractionResult, CostTracker]:
    """Extract cards using section-aware prompts with per-section model routing."""
    warn_threshold = config.cost_warn_at * config.cost_budget_limit
    cost_warned = False

    client = anthropic.Anthropic()
    all_cards: list[AnkiCard] = []
    model_used = ""

    # Section-level max_cards: use config value if smaller, else cap
    section_max_cards = min(config.cards_max_cards, _SECTION_MAX_CARDS)

    for section in sections:
        if not tracker.is_within_budget:
            logger.warning("Budget exceeded, stopping section processing")
            break

        if len(all_cards) >= config.cards_max_cards:
            logger.info(
                "Document card limit (%d) reached, stopping section processing",
                config.cards_max_cards,
            )
            break

        if not cost_warned and tracker.total_cost >= warn_threshold:
            logger.warning(
                "Cost approaching budget limit: $%.4f / $%.2f (%.0f%%)",
                tracker.total_cost,
                tracker.budget_limit,
                (tracker.total_cost / tracker.budget_limit) * 100,
            )
            cost_warned = True

        # Per-section model routing based on section size and section card count
        model = select_model(
            text_length=section.char_count,
            card_count=section_max_cards,
            force_model=config.model if config.model_overridden else None,
        )
        if not model_used:
            model_used = model

        user_prompt = build_section_prompt(
            section,
            document_title=document_title,
            max_cards=section_max_cards,
            card_types=config.cards_card_types,
            focus_topics=focus_topics,
            bloom_filter=bloom_filter,
            additional_tags=additional_tags,
        )

        messages: list[MessageParam] = [
            {"role": "user", "content": user_prompt}
        ]

        response = _call_with_retry(
            client=client,
            model=model,
            max_tokens=config.max_tokens,
            messages=messages,
        )

        if not response.content:
            logger.warning("Empty API response for section %s, skipping", section.id)
            continue

        first_block = response.content[0]
        if not hasattr(first_block, "text"):
            logger.warning(
                "Unexpected response block type for section %s: %s",
                section.id,
                type(first_block).__name__,
            )
            continue

        response_text = first_block.text
        cards = parse_cards_response(response_text)
        all_cards.extend(cards)

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
        cards=all_cards,
        model_used=model_used or DEFAULT_MODEL,
    )

    return result, tracker


def _call_with_retry(
    *,
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    messages: list[MessageParam],
) -> anthropic.types.Message:
    """Call Claude API with exponential backoff retry.

    Args:
        client: Anthropic client.
        model: Model ID.
        max_tokens: Max output tokens.
        messages: User messages.

    Returns:
        API response message.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            return _call_claude_api(
                client=client,
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
        except _RETRYABLE_ERRORS as e:
            last_error = e
            logger.warning(
                "API call attempt %d/%d failed: %s",
                attempt + 1,
                _MAX_RETRIES,
                e,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))

    raise RuntimeError(
        f"API call failed after {_MAX_RETRIES} retries: {last_error}"
    )
