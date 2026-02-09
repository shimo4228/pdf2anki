"""LLM critique functions for pdf2anki quality assurance.

Sends low-confidence cards to Claude for review and improvement,
parsing the structured JSON response to improve, split, or remove cards.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic
from pydantic import ValidationError

from pdf2anki.config import DEFAULT_MODEL
from pdf2anki.cost import CostRecord, CostTracker, estimate_cost
from pdf2anki.prompts import CRITIQUE_PROMPT
from pdf2anki.schemas import AnkiCard

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _call_critique_api(
    *,
    client: anthropic.Anthropic,
    model: str,
    cards_json: str,
    source_text: str,
) -> anthropic.types.Message:
    """Call Claude API for card critique.

    Args:
        client: Anthropic client.
        model: Model ID.
        cards_json: JSON string of cards to critique.
        source_text: Original source text for hallucination check.

    Returns:
        Claude API Message response.
    """
    user_content = (
        f"## Cards to Review\n\n{cards_json}\n\n"
        f"## Original Source Text\n\n{source_text[:3000]}"
    )
    return client.messages.create(
        model=model,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": CRITIQUE_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )


def _parse_critique_response(response_text: str) -> list[dict[str, Any]]:
    """Parse and validate the LLM critique response JSON."""
    text = response_text.strip()

    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse critique response: %s", e)
        return []

    if not isinstance(data, list):
        logger.warning("Critique response is not a list")
        return []

    validated: list[dict[str, Any]] = []
    for review in data:
        if not isinstance(review, dict):
            continue
        if "card_index" not in review or "action" not in review:
            logger.warning("Review missing required fields: %s", review)
            continue
        validated.append(review)

    return validated


def critique_cards(
    *,
    cards: list[AnkiCard],
    source_text: str,
    cost_tracker: CostTracker,
    model: str = DEFAULT_MODEL,
) -> tuple[list[AnkiCard], CostTracker]:
    """Send low-confidence cards to LLM for critique and improvement.

    Args:
        cards: Cards to critique.
        source_text: Original source text for hallucination checking.
        cost_tracker: Cost tracker for budget enforcement.
        model: Claude model to use for critique.

    Returns:
        Tuple of (improved cards, updated CostTracker).
    """
    if not cards:
        return [], cost_tracker

    cards_data = [card.model_dump() for card in cards]
    cards_json = json.dumps(cards_data, ensure_ascii=False, indent=2)

    try:
        client = anthropic.Anthropic()
        response = _call_critique_api(
            client=client,
            model=model,
            cards_json=cards_json,
            source_text=source_text,
        )
    except anthropic.APIError as e:
        logger.error("API error during critique: %s", e)
        return list(cards), cost_tracker

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
    cost_tracker = cost_tracker.add(record)

    if not response.content:
        logger.warning("Empty critique response")
        return list(cards), cost_tracker

    first_block = response.content[0]
    if not hasattr(first_block, "text"):
        logger.warning("Unexpected response block type")
        return list(cards), cost_tracker

    response_text: str = first_block.text
    reviews = _parse_critique_response(response_text)

    reviewed_indices: set[int] = set()
    result_cards: list[AnkiCard] = []

    for review in reviews:
        idx = review.get("card_index")
        action = review.get("action", "")
        improved = review.get("improved_cards")

        if idx is None or not isinstance(idx, int):
            continue

        reviewed_indices.add(idx)

        if action == "remove":
            continue

        if action in ("improve", "split") and isinstance(improved, list):
            for item in improved:
                try:
                    new_card = AnkiCard.model_validate(item)
                    result_cards.append(new_card)
                except (ValidationError, TypeError) as e:
                    logger.warning("Skipping invalid improved card: %s", e)
        else:
            if 0 <= idx < len(cards):
                result_cards.append(cards[idx])

    for i, card in enumerate(cards):
        if i not in reviewed_indices:
            result_cards.append(card)

    return result_cards, cost_tracker
