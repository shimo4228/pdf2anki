# Next Features: Deep Research & Implementation Plan

> Created: 2026-02-09
> Status: Ready for Deep Research Phase
> Context: OpenAI support rejected (see `provider-selection-root-cause-analysis.md`)

---

## Executive Summary

**Decision:** Skip OpenAI support. Invest 15 hours in 4 high-value features instead.

**Features (Priority Order):**
1. ✅ Japanese Tokenization Fix (1h) - Quick win, known bug
2. 🎯 Image-Aware Card Generation (8h) - Highest value, new capability
3. ⚡ Interactive Review TUI (6h) - UX improvement
4. 📈 Prompt Evaluation Framework (4h) - Quality improvement

**Total:** 15 hours vs 10 hours for OpenAI support (better ROI)

---

## Feature 1: Japanese Tokenization Fix 🐛

### Current Problem (from MEMORY.md)

```python
# src/pdf2anki/extract.py
CHARS_PER_TOKEN = 4  # ← English-based assumption

# Reality for Japanese text:
"これは日本語のテストです" (13 chars)
→ Current estimate: 13/4 = 3.25 tokens (WRONG)
→ Actual usage: 13/2.5 = 5.2 tokens (CORRECT)

# Impact on 正理の海.pdf (572K chars, mostly Japanese):
→ Current estimate: 143,000 tokens → $2.15
→ Actual usage: 228,800 tokens → $3.43
→ Error: 60% underestimate ($1.28 off)
```

### Proposed Solution

Implement language-aware tokenization:

```python
def estimate_tokens(text: str) -> int:
    """Estimate token count with CJK language support."""
    cjk_chars = sum(1 for c in text if is_cjk(c))
    other_chars = len(text) - cjk_chars

    # Language-specific ratios
    cjk_tokens = cjk_chars / 2.5      # Japanese/Chinese/Korean
    other_tokens = other_chars / 4.0  # English/Latin scripts

    return int(cjk_tokens + other_tokens)

def is_cjk(char: str) -> bool:
    """Check if character is CJK (Chinese/Japanese/Korean)."""
    code_point = ord(char)
    return any([
        0x4E00 <= code_point <= 0x9FFF,   # CJK Unified Ideographs
        0x3040 <= code_point <= 0x309F,   # Hiragana
        0x30A0 <= code_point <= 0x30FF,   # Katakana
    ])
```

### Research Questions

1. **Tokenizer accuracy:** Should we use tiktoken library for precise counting vs estimation?
2. **Performance:** Is character-by-character CJK detection fast enough for large PDFs?
3. **Mixed content:** How to handle PDFs with 50/50 Japanese/English content?
4. **Model differences:** Do Haiku/Sonnet have different tokenization ratios?

### Impact

- ✅ Accurate cost estimation (within 1% error)
- ✅ Correct chunk size calculation
- ✅ Better model selection (Haiku vs Sonnet threshold)
- ✅ Improved Batch API usage decisions

### Files to Modify

- `src/pdf2anki/extract.py` - Add `estimate_tokens()`, `is_cjk()`
- `src/pdf2anki/cost.py` - Update cost calculation logic
- `tests/test_extract.py` - Add tokenization tests

---

## Feature 2: Image-Aware Card Generation 🎯

### Current Gap

```python
# src/pdf2anki/schemas.py:25
class CardType(str, Enum):
    BASIC = "basic"
    CLOZE = "cloze"
    IMAGE_OCCLUSION = "image_occlusion"  # ← DEFINED BUT NOT IMPLEMENTED
```

**Current behavior:** Images in PDFs are completely ignored during extraction.

### Proposed Architecture

```
PDF Page with Images
    ↓
pymupdf: Detect images (page.get_images())
    ↓
Extract page as PNG (page.get_pixmap())
    ↓
Claude Multimodal API (vision + text)
    ↓
Generate image-aware cards:
  - Basic cards (describe this diagram)
  - Image occlusion cards (label parts)
  - Visual association cards (connect text to image)
```

### Example Output

**Input:** Buddhist philosophy PDF with diagram of Four Noble Truths

**Current (text-only):**
```
Q: What are the Four Noble Truths?
A: Suffering, Origin, Cessation, Path
```

**After implementation (image-aware):**
```
Card 1 (Basic + Image):
Q: What does this diagram represent?
A: The Four Noble Truths (苦諦・集諦・滅諦・道諦)
[Image: diagram embedded in card]

Card 2 (Image Occlusion):
Q: Identify the highlighted element in this diagram
A: 苦諦 (Suffering)
[Image: diagram with one part highlighted]

Card 3 (Visual Association):
Q: Which arrow represents the path from 集諦 to 滅諦?
A: The upper right arrow labeled "修行道"
[Image: diagram with arrows]
```

### Research Questions

1. **Image detection threshold:** What % of page area should be image to trigger vision API?
2. **Claude vision API:**
   - How to combine image + text in single prompt?
   - Cost implications (vision pricing vs text-only)?
   - Does prompt caching work with images?
3. **Image occlusion format:**
   - Anki's image occlusion format requirements?
   - How to encode masked regions in card data?
4. **Quality:** Does Claude's vision capability work well for:
   - Hand-drawn diagrams in scanned PDFs?
   - Dense Japanese text in images (vertical writing)?
   - Mathematical equations as images?
5. **Performance:** Should images be processed in Batch API or real-time only?

### Technical Challenges

1. **Image extraction quality:**
   - DPI settings (150? 300?)
   - Color vs grayscale
   - File size management

2. **Prompt design:**
   - How to instruct Claude to generate image occlusion cards?
   - Should we use separate prompts for image-heavy vs text-heavy pages?

3. **Card schema:**
   - Where to store image data? (base64 in JSON? external files?)
   - How to handle image references in TSV export?

### Files to Create/Modify

**New files:**
- `src/pdf2anki/vision.py` - Vision API integration
- `src/pdf2anki/image_extraction.py` - Image detection & extraction
- `tests/test_vision.py` - Vision tests

**Modified files:**
- `src/pdf2anki/extract.py` - Add image detection logic
- `src/pdf2anki/prompts.py` - Add vision-specific prompts
- `src/pdf2anki/structure.py` - Handle image-aware card generation
- `src/pdf2anki/cost.py` - Add vision API pricing

---

## Feature 3: Interactive Review TUI ⚡

### Current Workflow Problem

```
User runs: pdf2anki convert input.pdf -o cards.tsv
    ↓
47 cards generated
    ↓
ALL cards exported to cards.tsv
    ↓
User imports to Anki
    ↓
User manually deletes 12 low-quality cards in Anki
    ↓
User manually edits 8 cards with typos
```

### Proposed Workflow

```
User runs: pdf2anki convert input.pdf --review
    ↓
47 cards generated
    ↓
Rich TUI opens:

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Card Review: input.pdf (47 cards)         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Card 1/47                     [Quality: 8.5/10]
┌──────────────────────────────────────────────┐
│ Front: 四諦とは何か？                          │
│ Back:  苦諦・集諦・滅諦・道諦の4つの真理         │
│ Tags:  仏教, 基本概念                          │
└──────────────────────────────────────────────┘

[A]ccept [E]dit [R]eject [I]nfo [N]ext [P]rev
[F]ilter [S]ave [Q]uit

> a  (accept)
    ↓
Card 2/47 shown...
    ↓
User reviews all 47 cards
    ↓
Press [S]ave
    ↓
Only approved 35 cards exported to cards.tsv
```

### Features to Implement

1. **Card navigation:**
   - Next/Previous
   - Jump to card number
   - Filter by quality score (<7.0 only)
   - Filter by tag

2. **Card actions:**
   - Accept (✓)
   - Reject (✗)
   - Edit (inline editing of Front/Back/Tags)
   - View quality info (detailed critique)

3. **Bulk operations:**
   - Accept all above threshold
   - Reject all duplicates
   - Tag all in section

4. **Statistics panel:**
   - Total: 47 cards
   - Accepted: 35
   - Rejected: 12
   - Avg quality: 8.2/10

5. **Export options:**
   - Export accepted only
   - Export all with status flags
   - Save review session (resume later)

### Research Questions

1. **Rich TUI library:**
   - Best widgets for card display? (Table? Panel? Layout?)
   - How to handle inline editing in Rich?
   - Keyboard shortcuts best practices?

2. **UX design:**
   - Should quality scores be color-coded (red <6, yellow 6-8, green >8)?
   - Should there be undo functionality?
   - How to show diff when editing?

3. **Performance:**
   - How to efficiently load 500+ cards in TUI?
   - Should we paginate or keep all in memory?

4. **Persistence:**
   - Save review state to disk (resume interrupted session)?
   - File format for partial reviews?

### Files to Create

**New files:**
- `src/pdf2anki/tui/` - TUI module
  - `__init__.py`
  - `review.py` - Main review TUI
  - `widgets.py` - Custom Rich widgets
  - `keybindings.py` - Keyboard handler
- `tests/test_tui.py` - TUI tests (mock keyboard input)

**Modified files:**
- `src/pdf2anki/main.py` - Add `--review` flag
- `src/pdf2anki/convert.py` - Return cards object instead of writing immediately

---

## Feature 4: Prompt Evaluation Framework 📈

### Current Problem

```
Developer changes prompt in prompts.py
    ↓
Runs pdf2anki convert test.pdf
    ↓
Eyeballs the output
    ↓
"Hmm, seems okay?" (subjective, no data)
    ↓
Commits change (no validation)
```

### Proposed System

```
Developer creates eval dataset:

# evals/dataset.yaml
- id: "buddhism-01"
  text: "四諦とは、苦諦・集諦・滅諦・道諦の4つの真理である。"
  expected_cards:
    - front: "四諦とは何か"
      back: "苦諦・集諦・滅諦・道諦の4つの真理"
      tags: ["仏教", "基本"]
    - front: "四諦の4つを列挙せよ"
      back: "苦諦、集諦、滅諦、道諦"
      tags: ["仏教", "暗記"]

Developer runs eval:

$ pdf2anki eval --prompt prompts/v1.txt --dataset evals/dataset.yaml
    ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Evaluation Report: prompts/v1.txt    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Dataset: 10 texts, 23 expected cards

Recall:    17/23 (73.9%)  ← % of expected cards generated
Precision: 17/20 (85.0%)  ← % of generated cards are correct
F1 Score:  0.790
Quality:   8.2/10 avg
Duplicates: 2/20 (10%)
Cost:      $0.12

Per-item breakdown:
  buddhism-01: 2/2 cards ✓ (100%)
  buddhism-02: 1/2 cards ✗ (50% - missed cloze card)
  ...

Compare with v2:

$ pdf2anki eval --compare prompts/v1.txt prompts/v2.txt
    ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Comparison: v1 vs v2                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

               v1      v2      Diff
Recall:        73.9%   87.0%   +13.1% ✓
Precision:     85.0%   81.0%   -4.0%  ✗
Quality:       8.2     8.5     +0.3   ✓
Cost:          $0.12   $0.18   +50%   ✗

Recommendation: v2 has better coverage but higher cost.
Consider using v2 for comprehensive study, v1 for quick review.
```

### Evaluation Metrics

1. **Recall:** What % of expected cards were generated?
   - Formula: `correct_generated / expected_total`
   - Measures completeness

2. **Precision:** What % of generated cards are correct?
   - Formula: `correct_generated / generated_total`
   - Measures accuracy (no hallucinations)

3. **F1 Score:** Harmonic mean of recall & precision
   - Formula: `2 * (precision * recall) / (precision + recall)`

4. **Quality Score:** Average quality from critique pipeline

5. **Duplicate Rate:** % of generated cards that are duplicates

6. **Cost:** Total API cost for generation

### Research Questions

1. **Evaluation dataset:**
   - How many examples needed for statistical significance? (10? 50? 100?)
   - Should we create domain-specific datasets (Buddhism, Science, etc.)?
   - How to version datasets?

2. **Card matching algorithm:**
   - How to determine if a generated card matches an expected card?
   - Exact match? Semantic similarity? LLM-based comparison?
   - Should we use embeddings (cosine similarity)?

3. **Prompt variants:**
   - Should prompts be versioned in git?
   - File format: .txt? .md? Python templates?
   - How to parameterize prompts (model, max_tokens, etc.)?

4. **Continuous evaluation:**
   - Should eval run on every commit (CI integration)?
   - How to track regression?

5. **Advanced metrics:**
   - Should we measure Wozniak principles adherence?
   - Difficulty distribution (too easy? too hard?)?
   - Tag quality?

### Files to Create

**New files:**
- `src/pdf2anki/eval/` - Eval module
  - `__init__.py`
  - `dataset.py` - Dataset schema & loader
  - `metrics.py` - Recall, precision, F1, etc.
  - `matcher.py` - Card matching logic
  - `report.py` - Report generation
- `evals/` - Eval datasets
  - `dataset.yaml` - Main dataset
  - `buddhism.yaml` - Domain-specific
- `tests/test_eval.py` - Eval tests

**Modified files:**
- `src/pdf2anki/main.py` - Add `eval` subcommand

---

## Implementation Strategy

### Recommended Order

1. **Feature 1: Tokenization Fix** (1h)
   - Quick win
   - Unblocks accurate cost estimation for other features
   - No dependencies

2. **Feature 2: Image-Aware Cards** (8h)
   - Highest value
   - Most complex, needs deep research
   - New capability, not just improvement

3. **Feature 3: Interactive TUI** (6h)
   - UX improvement
   - Depends on Feature 2 (want to review image cards too)

4. **Feature 4: Eval Framework** (4h)
   - Quality improvement
   - Should be last so we can eval the improved prompts from Feature 2

### Research Phase Plan

For each feature, deep research should answer:

1. **Technical feasibility:** Can this be done with current tools?
2. **API capabilities:** Does Claude/pymupdf support this?
3. **Cost implications:** What's the $ impact?
4. **Design decisions:** What are the trade-offs?
5. **Implementation risks:** What could go wrong?

### Success Criteria

After implementation, the tool should:

- ✅ Handle image-rich PDFs (diagrams, charts, figures)
- ✅ Provide accurate cost estimates for Japanese text
- ✅ Allow user review before Anki import
- ✅ Enable data-driven prompt improvement

**Total value:** Much higher than OpenAI support with similar time investment.

---

## Next Steps

1. **Start new session for deep research**
2. **Research each feature in detail:**
   - Claude API capabilities (vision, pricing)
   - Rich TUI patterns
   - Eval dataset design
   - pymupdf image extraction
3. **Create implementation plan** with phases
4. **Begin implementation** starting with Feature 1

---

## References

- `docs/plans/provider-selection-root-cause-analysis.md` - Why OpenAI was rejected
- `MEMORY.md` - Project context
- `src/pdf2anki/schemas.py:25` - IMAGE_OCCLUSION type defined
- `src/pdf2anki/extract.py` - Current tokenization logic
