# Should We Add OpenAI Support? (Root Cause Analysis)

> Author: architect agent
> Date: 2026-02-09
> Triggered by: User identified logical contradiction in previous architecture decision

---

## The Brutal Truth

**You caught a real contradiction, and it reveals something important.**

The previous analysis said "skip the OpenAI Batch API because $5 savings is not worth 200-300 LOC" and then immediately recommended "implement the full OpenAI provider for 750 LOC to use $10 credit." The math does not work. If $5 is not worth 250 lines, then $10 is not worth 750 lines either. The per-dollar cost of code is nearly identical:

| Feature | LOC | Dollar Value | LOC per Dollar |
|---------|-----|-------------|----------------|
| OpenAI Batch API | 250 | $5 savings | 50 LOC/$ |
| Full OpenAI Provider | 750 | $10 credit | 75 LOC/$ |

The Batch API was actually the *better* deal per dollar, yet it was rejected. This inconsistency means the $10 credit was never the real justification. It was a rationalization.

**The tool works.** It has 456 tests, 96% coverage, mypy --strict, and it does its job: PDF to Anki cards via Claude. It processed a 572K-character Japanese PDF successfully. There is no user complaint, no missing feature blocking real work, no performance bottleneck. The tool is finished for its current purpose.

Adding OpenAI support is solving a problem that does not exist.

---

## ROI Analysis

### Investment (precise, based on codebase review)

The 750 LOC estimate from the previous analysis was optimistic. Here is a more honest accounting:

| Work Item | LOC | Hours |
|-----------|-----|-------|
| `providers/protocol.py` + `LLMResponse` | 50 | 0.5 |
| `providers/anthropic.py` (extract from structure.py + quality.py) | 150 | 1.5 |
| `providers/openai.py` (new implementation) | 120 | 1.5 |
| `providers/__init__.py` (factory + runtime checks) | 40 | 0.3 |
| Refactor `structure.py` (remove anthropic coupling) | 80 delta | 1.0 |
| Refactor `quality.py` (remove anthropic coupling) | 40 delta | 0.5 |
| Refactor `config.py` + `cost.py` (OpenAI pricing, provider field) | 60 | 0.5 |
| Refactor `main.py` (--provider flag, plumbing) | 30 | 0.3 |
| **Test migration** (209 mock sites across 5 files) | **200+** | **2.0** |
| New tests for OpenAI provider | 150 | 1.0 |
| Documentation | 100 | 0.5 |
| **Total** | **~1,000+** | **~9-10** |

The 4-6 hour estimate in the handoff document underestimates test migration. The 209 mock/patch calls in `tests/test_structure.py` (61), `tests/test_main.py` (55), `tests/test_e2e.py` (40), `tests/test_batch.py` (31), and `tests/test_quality.py` (22) each mock Anthropic internals at a low level (`_call_claude_api`, `anthropic.Anthropic()`, response object chains). Migrating these to provider-level mocks is not a trivial find-and-replace -- each test needs individual attention to ensure the mock still validates the right behavior.

### Return

| Return | Dollar Value | Certainty |
|--------|-------------|-----------|
| Use $10 OpenAI credit | $10 | High (credit exists) |
| Cost savings from cheaper OpenAI models | $0-5/month | Low (usage is sporadic) |
| Provider flexibility for future | $0 (speculative) | Very Low |
| Portfolio/learning value | $0 (non-monetary) | Medium |

**Total quantifiable return: $10-15 one-time.**

### Verdict

**Investment of ~1,000 LOC and ~10 hours to capture ~$10 of value is objectively poor ROI.** This is equivalent to paying yourself roughly $1/hour in credit recovery. Even accounting for learning value, any of the alternative investments listed below would produce more tangible benefit.

---

## Real Motivation Assessment

### The stated reason: "$10 OpenAI credit"

This is an anchoring effect. The $10 credit creates a feeling of waste ("I'm losing money by not using it"), which triggers action. But $10 that expires unused is a $10 loss. $10 that requires 10 hours of work to capture is actually a net loss when you account for the opportunity cost of those 10 hours.

### The likely real reasons (in order of probability)

**1. Technical curiosity.** "How does OpenAI's structured outputs compare to Anthropic's? Can my prompts work with GPT-4o?" This is a legitimate learning interest, but it does not require 1,000 LOC of production code. A 30-line throwaway script answers this question.

**2. Architecture appeal.** "A clean provider abstraction layer would make this codebase more elegant." This is the most dangerous motivation because it feels productive while producing negative value. The current codebase is already clean -- 12 files, each under 700 lines, 96% coverage, mypy --strict. Adding an abstraction layer that only has one real consumer (Anthropic) and one token consumer (OpenAI, used once to burn $10) makes the architecture *worse*, not better. Abstractions earn their keep through use; unused abstractions are dead weight.

**3. Sunk cost of research.** Two planning documents already exist (`provider-selection-handoff.md`, `provider-selection-architecture-briefing.md`). An architecture decision document was just written. There is psychological pressure to justify this invested effort by proceeding to implementation. This is the textbook sunk cost fallacy.

**4. Completionism.** "A tool that only supports one provider feels incomplete." But the tool's description in `pyproject.toml` line 3 literally says "using Claude AI." Single-provider is the design, not a limitation.

### Does it justify the work?

**No.** None of these motivations justify 1,000 LOC and 10 hours of production-quality engineering (with tests, types, documentation). Curiosity can be satisfied with a spike. Architecture appeal is a trap. Sunk cost should be ignored. Completionism is not a user need.

---

## Alternative Investments (What Else Could Be Built)

With the same 10 hours, here are features that would deliver more value to the tool's actual purpose -- generating high-quality Anki flashcards:

### 1. Image-aware card generation (~8 hours, HIGH value)

**The problem:** The project defines an `image_occlusion` card type in `src/pdf2anki/schemas.py` (line 25), but there is no implementation. PDFs with diagrams, charts, and figures currently lose all visual information during text extraction.

**The solution:** Use Claude's vision capability to process PDF page images directly. When `pymupdf4llm` extracts a page with significant image content, send the page image to Claude's multimodal API alongside the text. Generate cards that reference visual elements ("What does this diagram show?", image occlusion cards).

**Why this is better:** This is a genuine capability gap in the current tool. Japanese academic texts (like the 572K-character "Shouri no Umi" PDF mentioned in the research docs) often contain diagrams, tables, and figures that are completely lost in text-only extraction. This feature makes the tool useful for a category of documents it currently cannot handle.

### 2. Interactive card review/edit TUI (~6 hours, MEDIUM-HIGH value)

**The problem:** Generated cards go directly to TSV/JSON. There is no way to review, filter, or edit cards before importing to Anki. The quality pipeline catches some issues automatically, but the user has no input.

**The solution:** A Rich-based TUI (the project already depends on Rich) that shows generated cards in a table, allows the user to approve/reject/edit each card, and then exports only approved cards.

**Why this is better:** This directly improves the user experience of the tool's core workflow. Every card generation session would benefit, not just the hypothetical sessions using OpenAI.

### 3. Prompt tuning and evaluation framework (~4 hours, MEDIUM value)

**The problem:** The system prompt in `src/pdf2anki/prompts.py` is a single static string. There is no way to compare prompt variants, no eval harness, and no ground truth dataset to measure card quality objectively.

**The solution:** Create a small eval framework: a set of reference texts with expected cards, a scoring function that compares generated cards against expectations, and a simple CLI to run prompt variants and compare results.

**Why this is better:** Prompt quality is the single biggest lever on output quality. A 10% improvement in the prompt produces 10% better cards for every user, every document, every run. An OpenAI provider produces zero improvement for 95%+ of runs.

### 4. Fix the Japanese tokenization estimate (~1 hour, LOW but correct)

**The problem:** Documented in MEMORY.md: "`CHARS_PER_TOKEN = 4` in extract.py is inaccurate for JP-heavy docs." The actual ratio for Japanese is ~2-3 chars/token. This means cost estimates are wrong and chunk sizes may be suboptimal.

**Why this is better:** It is a genuine known bug that affects the tool's core accuracy. It takes 1 hour and fixes a real issue versus 10 hours to add a feature nobody asked for.

### Ranking

| Feature | Hours | Value to Core Mission | Frequency of Benefit |
|---------|-------|----------------------|---------------------|
| Image-aware cards | 8 | HIGH (new capability) | Every PDF with images |
| Interactive TUI | 6 | MEDIUM-HIGH (UX) | Every session |
| Prompt eval framework | 4 | MEDIUM (quality) | Every prompt iteration |
| JP tokenization fix | 1 | LOW (accuracy) | Every JP document |
| **OpenAI provider** | **10** | **NONE** (lateral move) | **Rarely/never** |

Every single alternative provides more value than OpenAI support.

---

## The Sunk Cost Trap

Three documents have been written about this feature:

1. `docs/plans/provider-selection-handoff.md` -- handoff doc
2. `docs/plans/provider-selection-architecture-briefing.md` -- briefing for architect
3. The previous architecture decision (session output)

Plus this current analysis. That is roughly 2-3 hours of planning for a feature that should not be built. The planning itself has become a sunk cost that creates pressure to proceed.

**The right response to sunk cost is to ignore it.** The question is not "should we justify the research we already did?" but "if we were starting fresh today with no prior research, would we prioritize OpenAI support over image-aware cards or a review TUI?" The answer is obviously no.

---

## Honest Scenarios

**Will you actually use OpenAI after implementing it?**

Based on the evidence:
- The project is described as "using Claude AI" everywhere
- The model routing (`cost.py` lines 115-140) is hardcoded to Claude models (Haiku/Sonnet)
- The Batch API (which provides 50% savings) only works with Anthropic
- The prompt caching (which reduces input costs) only works with Anthropic
- The quality pipeline critique uses Claude

If you switch to OpenAI, you lose prompt caching, Batch API 50% discount, and the model routing intelligence. You would be paying more for a potentially worse experience, just to use $10 of credit.

**Realistic usage prediction:** You would use OpenAI once or twice to see the output, compare it with Claude's output, conclude that Claude is better for Japanese text (Claude's Japanese capability is widely regarded as superior), and never use OpenAI again. The provider abstraction would then sit in the codebase as dead complexity, maintained forever.

---

## Recommended Action

### Option C: Skip Entirely

**Do not build OpenAI support.**

The $10 credit is not a problem to solve. It is a distraction. Use it for something else (a different project, API experimentation, a throwaway comparison script) or let it expire. $10 is less than the electricity cost of the compute time needed to implement this feature.

**Invest the time in image-aware card generation instead.** This is a genuine capability gap with clear user value. It uses Claude's multimodal API (which you are already paying for), works with the existing pipeline, and makes the tool useful for an entire category of documents it currently cannot handle.

### Rationale

1. **No user need.** The tool works. Nobody is asking for OpenAI support.
2. **Negative ROI.** $10 return on 10 hours of work is below minimum wage in any country.
3. **Opportunity cost.** Image-aware cards, interactive review, and prompt evaluation are all higher value.
4. **Maintenance burden.** The provider abstraction adds permanent complexity for a feature that will be used once.
5. **YAGNI.** "You Aren't Gonna Need It" exists precisely for this situation. If a real need for OpenAI support emerges in the future (multiple users requesting it, Anthropic pricing becomes uncompetitive, Claude's Japanese quality degrades), the codebase is small enough (3,348 lines) that the provider abstraction can be added in a day.

---

## If You Still Want To Proceed Despite My Recommendation

If you decide to proceed anyway (perhaps for learning value, and that is a valid personal reason even if it is not a valid engineering reason), here are guardrails:

### Hard constraints

1. **Option B only: Minimal Spike.** Write a throwaway script (not production code) that sends the existing system prompt to OpenAI's API with structured outputs and compares the card quality. 50-100 lines, no tests, no abstraction layer. Save it as `scripts/openai_spike.py`. This satisfies curiosity at 1/10th the cost.

2. **Time-box to 2 hours maximum.** If it takes longer than 2 hours, stop. The learning value diminishes rapidly after the initial "does it work?" question is answered.

3. **Do not refactor the existing codebase.** The provider abstraction refactoring (migrating 209 mock sites, changing function signatures across 5 files) is the most expensive part of the work and provides zero value unless you are genuinely going to use OpenAI in production.

4. **Do not merge to main.** Keep it on a branch or in `scripts/`. If you decide later that you want production OpenAI support, the spike will inform the design. If not, delete the branch.

### What the spike should answer

- Does GPT-4o follow the Wozniak principles prompt as well as Claude?
- How does the structured output quality compare for Japanese text?
- Are there parsing differences that would require changes to `parse_cards_response()`?
- What is the actual cost per document compared to Claude with Batch API?

If the spike reveals that OpenAI produces meaningfully better cards for certain document types, then and only then should you consider production implementation.

---

## Closing Thought

The project's own design philosophy says: **"Don't Import the Warehouse for a Single Wheel."**

The same principle applies to features. Do not build the provider abstraction warehouse for a single $10 wheel. The best code is the code you do not write.

---

## Decision: SKIP OpenAI Support

**User Decision (2026-02-09):** Skip OpenAI support entirely. Invest time in the 4 high-value features instead.

**Next Steps:** Deep research on the 4 alternative features, then create implementation plan.
