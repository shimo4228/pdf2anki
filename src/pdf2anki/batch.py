"""Batch API processing for pdf2anki.

Submits multiple section-level card generation requests to Anthropic's
Message Batches API for 50% cost savings. Includes polling with timeout
and result parsing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import anthropic

from pdf2anki.config import AppConfig
from pdf2anki.cost import select_model
from pdf2anki.prompts import SYSTEM_PROMPT, build_section_prompt
from pdf2anki.schemas import AnkiCard
from pdf2anki.section import Section
from pdf2anki.structure import parse_cards_response

logger = logging.getLogger(__name__)

_SECTION_MAX_CARDS = 20  # Per-section card cap (matches structure.py)


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """A single request within a message batch. Immutable."""

    custom_id: str
    model: str
    user_prompt: str
    system_prompt: str
    max_tokens: int


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Parsed result from a single batch entry. Immutable."""

    custom_id: str
    cards: list[AnkiCard]
    input_tokens: int
    output_tokens: int
    model: str


def create_batch_requests(
    sections: list[Section],
    *,
    document_title: str,
    config: AppConfig,
    focus_topics: list[str] | None = None,
    bloom_filter: list[str] | None = None,
    additional_tags: list[str] | None = None,
) -> list[BatchRequest]:
    """Create batch requests from sections with per-section model routing.

    Args:
        sections: Document sections to process.
        document_title: Document title for prompt context.
        config: Application configuration.
        focus_topics: Topics to emphasize.
        bloom_filter: Bloom levels to include.
        additional_tags: Extra tags for all cards.

    Returns:
        List of BatchRequest objects ready for submission.
    """
    if not sections:
        return []

    section_max_cards = min(config.cards_max_cards, _SECTION_MAX_CARDS)
    requests: list[BatchRequest] = []

    for section in sections:
        model = select_model(
            text_length=section.char_count,
            card_count=section_max_cards,
            force_model=config.model if config.model_overridden else None,
        )

        user_prompt = build_section_prompt(
            section,
            document_title=document_title,
            max_cards=section_max_cards,
            card_types=config.cards_card_types,
            focus_topics=focus_topics,
            bloom_filter=bloom_filter,
            additional_tags=additional_tags,
        )

        requests.append(
            BatchRequest(
                custom_id=section.id,
                model=model,
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=config.max_tokens,
            )
        )

    return requests


def submit_batch(
    requests: list[BatchRequest],
    *,
    client: anthropic.Anthropic | None = None,
) -> str:
    """Submit batch requests to the Anthropic Message Batches API.

    Args:
        requests: List of BatchRequest objects to submit.
        client: Anthropic client (created if None).

    Returns:
        Batch ID string.

    Raises:
        ValueError: If requests list is empty.
    """
    if not requests:
        raise ValueError("Cannot submit an empty batch request list")

    if client is None:
        client = anthropic.Anthropic()

    from anthropic.types.messages import batch_create_params

    api_requests: list[batch_create_params.Request] = []
    for req in requests:
        api_requests.append({
            "custom_id": req.custom_id,
            "params": {
                "model": req.model,
                "max_tokens": req.max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": req.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [
                    {"role": "user", "content": req.user_prompt}
                ],
            },
        })

    batch = client.messages.batches.create(requests=api_requests)
    logger.info("Batch submitted: %s (%d requests)", batch.id, len(requests))
    return batch.id


def poll_batch(
    batch_id: str,
    *,
    client: anthropic.Anthropic | None = None,
    poll_interval: float = 30.0,
    timeout: float = 3600.0,
) -> object:
    """Poll until a batch completes processing.

    Args:
        batch_id: The batch ID to poll.
        client: Anthropic client (created if None).
        poll_interval: Seconds between polls.
        timeout: Maximum seconds to wait before raising TimeoutError.

    Returns:
        The completed batch object.

    Raises:
        TimeoutError: If batch does not complete within timeout.
    """
    if client is None:
        client = anthropic.Anthropic()

    start = time.monotonic()

    while True:
        batch = client.messages.batches.retrieve(batch_id)

        if batch.processing_status == "ended":
            logger.info("Batch %s completed", batch_id)
            return batch

        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            raise TimeoutError(
                f"Batch {batch_id} did not complete within "
                f"{timeout:.0f}s (status: {batch.processing_status})"
            )

        logger.debug(
            "Batch %s status: %s (%.0fs elapsed)",
            batch_id,
            batch.processing_status,
            elapsed,
        )
        time.sleep(poll_interval)


def collect_batch_results(
    batch_id: str,
    *,
    client: anthropic.Anthropic | None = None,
) -> list[BatchResult]:
    """Retrieve and parse results from a completed batch.

    Args:
        batch_id: The completed batch ID.
        client: Anthropic client (created if None).

    Returns:
        List of BatchResult objects with parsed cards.
    """
    if client is None:
        client = anthropic.Anthropic()

    results: list[BatchResult] = []

    for entry in client.messages.batches.results(batch_id):
        if entry.result.type != "succeeded":
            logger.warning(
                "Batch entry %s failed: %s",
                entry.custom_id,
                entry.result.type,
            )
            continue

        message = entry.result.message
        if not message.content:
            logger.warning(
                "Batch entry %s returned empty content", entry.custom_id
            )
            continue

        first_block = message.content[0]
        if not hasattr(first_block, "text"):
            logger.warning(
                "Batch entry %s: unexpected block type: %s",
                entry.custom_id,
                type(first_block).__name__,
            )
            continue

        try:
            cards = parse_cards_response(first_block.text)
        except ValueError:
            logger.warning(
                "Batch entry %s: failed to parse response as cards",
                entry.custom_id,
            )
            continue

        results.append(
            BatchResult(
                custom_id=entry.custom_id,
                cards=cards,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                model=message.model,
            )
        )

    return results
