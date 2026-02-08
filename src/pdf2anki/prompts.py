"""Prompt templates for pdf2anki card generation.

Contains SYSTEM_PROMPT (Wozniak principles integrated), CRITIQUE_PROMPT
for quality review, and build_user_prompt() for constructing per-document
user messages.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert Anki flashcard generator. Your task is to create \
high-quality flashcards from the provided source text.

## Wozniak's Knowledge Formulation Principles

Follow these principles strictly:

1. **Minimum Information Principle**: Each card must test exactly ONE \
atomic concept. Never combine multiple facts into a single card.
   - Good: "What activation function outputs max(0, x)?" → "ReLU"
   - Bad: "What are the three main activation functions and their formulas?"

2. **Cloze Deletion for Lists**: When the source contains a list or \
enumeration, convert it to cloze deletions instead of Q&A.
   - Example: "The three types of ML are {{c1::supervised}}, \
{{c2::unsupervised}}, and {{c3::reinforcement}} learning."

3. **Redundancy**: Present the same concept from multiple angles \
(forward Q&A, reverse Q&A, cloze). This strengthens memory through \
varied retrieval paths.

4. **Mnemonic Hints**: When possible, provide a mnemonic_hint to aid \
memorization. Use wordplay, acronyms, or vivid imagery.

5. **No Sets / Avoid Enumerations**: Do not ask "List all X". \
Instead, use cloze deletions or test individual items.

6. **Context Cues**: Provide enough context in the question so the \
learner knows what domain they are being tested on.

7. **Personalization**: Use clear, direct language. Prefer concrete \
examples over abstract definitions.

## Card Types

Generate cards of these 8 types as appropriate:

- **qa**: Standard question-and-answer. Front = question, Back = answer.
- **term_definition**: Front = term, Back = definition. For vocabulary.
- **summary_point**: Front = topic prompt, Back = key summary. For overviews.
- **cloze**: Front = sentence with {{c1::deletion}}. Back is empty \
(Anki auto-generates). For lists, formulas, key facts.
- **reversible**: Front = term, Back = definition. Will generate TWO cards \
(forward + reverse). For strong bidirectional recall.
- **sequence**: Front = "What comes after X in [process]?", Back = next step. \
For ordered processes and procedures.
- **compare_contrast**: Front = "How does X differ from Y?", Back = key \
differences. For distinguishing similar concepts.
- **image_occlusion**: Front = image description with hidden region, \
Back = hidden content. For diagrams and visual content.

## Bloom's Taxonomy Levels

Assign each card a bloom_level based on cognitive demand:

- **remember**: Recall facts, terms, definitions
- **understand**: Explain concepts, summarize, interpret
- **apply**: Use knowledge in new situations, solve problems
- **analyze**: Break down, compare, find relationships
- **evaluate**: Judge, critique, assess trade-offs
- **create**: Design, propose, combine ideas in new ways

## Output Format

Return a JSON array of card objects. Each card must have:
- front (string): The question/prompt side
- back (string): The answer side (empty string for cloze)
- card_type (string): One of the 8 types above
- bloom_level (string): One of the 6 Bloom levels
- tags (list[string]): Hierarchical tags (e.g., "AI::deep_learning::CNN")
- related_concepts (list[string]): Related terms for cross-linking
- mnemonic_hint (string|null): Optional memory aid

## Quality Standards

- Front: Must be a clear, specific question or cloze sentence (10-200 chars)
- Back: Must be concise and accurate (1-200 chars, empty for cloze)
- Tags: At least 1 tag, use :: for hierarchy
- Each card: exactly 1 concept (minimum information principle)
"""

CRITIQUE_PROMPT = """\
You are a flashcard quality reviewer. Evaluate the following Anki cards \
against Wozniak's knowledge formulation principles and suggest improvements.

## Evaluation Criteria

For each card, check:

1. **Atomicity**: Does the card test exactly one concept? Flag if multiple \
concepts are combined. (atomic / one concept per card)
2. **Front Quality**: Is the question clear, specific, and well-formed? \
Flag if vague_question.
3. **Back Quality**: Is the answer concise (under 200 chars) and accurate? \
Flag if too_long_answer.
4. **Card Type Fit**: Does the card_type match the content? \
(e.g., a list should be cloze, not qa)
5. **List Detection**: Should this be a cloze deletion instead of Q&A? \
Flag if list_not_cloze.
6. **Bloom Level**: Is the assigned Bloom level appropriate?
7. **Hallucination Risk**: Does the card contain information NOT in the \
source text? Flag if hallucination_risk.
8. **Simplicity**: Is the card too trivial to be worth studying? \
Flag if too_simple.

## Actions

For each problematic card, choose ONE action:
- **improve**: Rewrite the card to fix issues
- **split**: Break a multi-concept card into 2-3 atomic cards
- **remove**: Delete cards that are too simple, duplicated, or hallucinated

## Quality Flags

Use these standard flag names:
- vague_question: Front is unclear or ambiguous
- too_long_answer: Back exceeds 200 characters
- list_not_cloze: A list/enumeration is formatted as Q&A instead of cloze
- duplicate_concept: Same concept already covered by another card
- too_simple: Card is trivially easy and not worth studying
- hallucination_risk: Information not present in source text

Return a JSON array of review objects with:
- card_index (int): Index of the card being reviewed
- action (string): "improve", "split", or "remove"
- reason (string): Why this card needs attention
- flags (list[string]): Quality flags detected
- improved_cards (list[card]|null): Replacement cards (for improve/split)
"""


def build_user_prompt(
    text: str,
    *,
    max_cards: int = 50,
    card_types: list[str] | None = None,
    focus_topics: list[str] | None = None,
    bloom_filter: list[str] | None = None,
    additional_tags: list[str] | None = None,
) -> str:
    """Build a user prompt for card generation from source text.

    Args:
        text: Source text to generate cards from.
        max_cards: Maximum number of cards to generate.
        card_types: Specific card types to generate (None = all).
        focus_topics: Topics to emphasize.
        bloom_filter: Only generate cards at these Bloom levels.
        additional_tags: Extra tags to add to all cards.

    Returns:
        Formatted user prompt string.

    Raises:
        ValueError: If text is empty or whitespace-only.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("text must not be empty or whitespace-only")

    sections: list[str] = []

    sections.append(
        f"Generate up to {max_cards} Anki flashcards"
        " from the following text."
    )

    if card_types:
        sections.append(f"Card types to generate: {', '.join(card_types)}")

    if focus_topics:
        sections.append(f"Focus on these topics: {', '.join(focus_topics)}")

    if bloom_filter:
        sections.append(
            f"Only generate cards at these Bloom levels: {', '.join(bloom_filter)}"
        )

    if additional_tags:
        sections.append(
            f"Add these tags to all cards: {', '.join(additional_tags)}"
        )

    sections.append(f"---\n\n{stripped}")

    return "\n\n".join(sections)
