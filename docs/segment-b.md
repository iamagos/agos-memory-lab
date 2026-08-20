# Segment B: the baseline ladder

Segment B asks one question: **does routing a conversation history through
`agos-memory` produce a better reader context than retrieving raw sessions, at
equal cost to the reader?**

It is the first segment that spends money, and the first that can produce a
number anyone would want to quote. Both facts argue for settling the design
before any call is made, because most ways of running this experiment measure
something other than what they claim to.

Read `docs/segment-a.md` first. Segment B inherits its frozen inputs and its
discipline: nothing is paid for until the credential-free part has been run and
inspected.

## What Segment B claims, and what it does not

**Claims.** On a fixed 30-case stratified sample of LongMemEval-S, holding
reader, judge, prompts, and context budget identical, a ranking of acquisition
and retrieval strategies by answer accuracy, with an explicit accuracy-versus-
context-budget curve for each.

**Does not claim.** A LongMemEval leaderboard score. A comparison against mem0
or any other memory product. A result with publication-grade statistical power —
see [Statistical power](#statistical-power), which is the most important
limitation in this document.

## Frozen inputs

Identical to Segment A, and unchanged for the life of the segment:

- Benchmark LongMemEval `9e0b455f4ef0e2ab8f2e582289761153549043fc`.
- Cleaned dataset `98d7416c24c778c2fee6e6f3006e7a073259d48f`.
- Case manifest `manifests/longmemeval-s-balanced-30-v1.json`,
  `sha256:3806648c2bb691a59bec7df30cc611dcbea34aefdc7e9841b433a4a0ec103a14`.
- 30 cases; 26 eligible after excluding 4 abstention cases, which are scored
  separately and never folded into the headline number.

Any change to these invalidates every receipt in the segment.

## The ladder

Each rung changes exactly one acquisition or retrieval choice. Reader, judge,
prompts, and budget are held fixed across all of them.

| Rung | Lane | Credentials | Role |
| --- | --- | --- | --- |
| 0 | oracle evidence | none for retrieval | Debugging ceiling. **Not a comparable lane** — it reads the gold answer sessions. Use it to detect reader failure, never to claim a result. |
| 1 | full raw history | none for retrieval | Official no-retrieval control. See the context-window constraint below. |
| 2 | raw BM25 (`lexical`) | none for retrieval | Primary credential-free control. Already characterised by Gate 4. |
| 3 | raw dense (`qdrant-dense`) | none for retrieval | Embedding retrieval, local Qdrant + FastEmbed. |
| 4 | raw hybrid (`qdrant-hybrid`) | none for retrieval | Lexical and dense planes combined. |
| 5 | extracted AGOS memory (`--source memories`) | **extraction is paid** | The treatment. Requires Segment A Gate 5 first. |

Rungs 1-4 need no credentials for the retrieval half; only the reader and judge
cost money. Rung 5 additionally requires a paid extraction pass over the corpus.

Rungs 3 and 4 run through the PEP 723 script header, not the project venv:

```powershell
uv run longmem.py run --retriever qdrant-dense ...
```

`.venv\Scripts\python.exe` raises `qdrant_dependency_missing:run_with_uv_script`
for those two lanes. The embedding model is already cached under `data/models/`,
so they are offline; budget roughly 30 seconds per case.

## Budget parity: the central design decision

This is the part that is easy to get wrong, and getting it wrong invalidates the
whole segment.

The raw lanes select **whole sessions**, which measured at roughly 12,500
characters each. The memory lane selects **extracted memories**, which are
one or two orders of magnitude smaller. So "give both lanes the same budget"
means two completely different experiments depending on how budget is denominated:

- **Equal item budget** (`--top-k`) hands the raw lanes vastly more text for the
  same nominal budget. It flatters raw retrieval and is indefensible.
- **Equal character budget** (`--chars`) lets the memory lane fit far more items
  into the same reader context. This is the honest one, because characters are
  what the reader actually pays for.

**Decision: freeze `--chars`, and make it the only budget that binds.**

Gate 4 demonstrated why this needs saying out loud. Its defaults were
`--top-k 10 --chars 180000`, and the kernel selected exactly 10 sessions on
every case while using only 123,318 characters — a mean 68% of the cap. The run
was **item-bound, not character-bound**, and the item cap was doing invisible
work. Every answer session it dropped is recorded in `kernel.outcomes` with
`reason: item_budget`.

To prevent a hidden second control, every Segment B run sets:

- `--top-k 100` — the maximum, so it never binds.
- `--candidates 100` — the maximum, so the candidate pool is not a differing
  control between lanes.
- `--chars <B>` — the single swept budget.

### The budget grid

Gate 4's follow-up sweep showed answer-evidence coverage is a strong, near-linear
function of the character budget, which means any **single-point** comparison is
a budget artifact waiting to happen. Segment B therefore reports curves.

Grid, anchored on the Gate 4 default and doubling either side:

| Budget | `--chars` | Approx. reader tokens |
| --- | ---: | ---: |
| B1 | 45,000 | ~11,000 |
| B2 | 90,000 | ~22,000 |
| B3 | 180,000 | ~45,000 |
| B4 | 360,000 | ~90,000 |

Token figures assume ~4 characters per token and are for planning only; the
receipts record actual usage.

### The full-history control has a context-window problem

Measured on all 30 manifest cases with the caps lifted, the complete haystack is:

| Statistic | Value |
| --- | ---: |
| Mean characters per case | 498,276 |
| Median | 498,336 |
| **Maximum** | **510,501** |
| Mean sessions per case | 47.9 |
| Cases truncated | 0 of 30 |

The largest case is roughly **127,600 tokens** before the prompt and question are
added. A 128k-context reader cannot hold it. Either pick a reader with a context
window comfortably above 128k, or run rung 1 at an explicit character cap and
report it as a *truncated* full-history control — never as "full history"
unqualified.

## Free pre-flight: bound every lane before paying for any

Answer-evidence coverage — whether the session containing the answer survives
into the selected context — is computable with **no model call at all**, from the
`longmem.py run` receipt alone. It is a hard ceiling on what the reader can
possibly get right.

So the first phase of Segment B costs nothing:

1. Run every (lane, budget) cell credential-free. That is 4 retrieval lanes x 4
   budgets, plus oracle and full history: 18 runs.
2. For each cell compute full / partial / missing answer-session coverage over
   the 26 eligible cases.
3. Inspect the `reason` field on every omission, exactly as the Gate 4 follow-up
   did. A `char_budget` omission is the experiment working as designed. Anything
   else is a bug to fix before spending.

Only then choose which cells are worth a reader call. Cells whose coverage is
identical will almost certainly produce near-identical accuracy, and paying to
confirm that is a poor use of the budget.

This phase is also the honest place to discover that a lane is broken, and it is
free to repeat.

## Measurement

- **Primary.** Judge-scored accuracy over the 26 eligible cases, per lane, per
  budget point, plotted against measured input tokens.
- **Secondary.** Abstention accuracy over the 4 abstention cases, reported
  separately and never averaged into the primary number.
- **Descriptive.** Answer-evidence coverage, item counts, and characters
  selected. Item counts are reported, never controlled.
- **Per stratum.** All six question types, because Gate 4 showed they behave very
  differently — `single-session-user` scored a perfect 1.000 on kernel recall
  while `single-session-preference` scored 0.200 at rank 1.

Reader and judge are pinned model identities with hard cost caps, run through
`qa.py read` and `qa.py judge`. Receipts are joined by `compare.py`, which makes
no model call and performs no retrieval — it only joins completed receipts, so
the comparison itself cannot silently introduce a difference.

## Cost model

The measured token volumes are the reliable part; prices are not, and must be
filled in from the actual deployment before anything is authorised.

Per lane, per budget point, reader input tokens ≈ `30 x B_tokens`:

| Budget | Reader input tokens per lane |
| --- | ---: |
| B1 (45k chars) | ~340,000 |
| B2 (90k chars) | ~675,000 |
| B3 (180k chars) | ~1,350,000 |
| B4 (360k chars) | ~2,700,000 |

Four retrieval lanes across the full grid is therefore ~20.3M reader input
tokens, plus ~3.7M for the single full-history control. Reader output is small
(~200 tokens per answer). The judge sees only question, hypothesis, and
reference — on the order of 1,000 tokens per case.

Two consequences worth internalising before choosing a reader:

- The **reader** dominates total cost, and it scales linearly with the budget
  grid. Dropping B4 nearly halves the segment.
- The **extractor** for rung 5 processes the whole haystack — 14.9M characters,
  roughly 3.7M input tokens — once. It should be a small model, which is
  convenient, because Segment C sweeps exactly that choice.

Every call already carries `--max-cost`, which is the real-time guard. Azure
budget alerts fire on actual spend and lag by hours; they are not a control.

## Statistical power

**This is the limitation that matters most, and it should appear in any writeup
of the results.**

There are 26 eligible paired cases. For a paired accuracy comparison between two
lanes, McNemar's exact test needs at least **six discordant pairs, all favouring
the same lane**, to reach p < 0.05 two-sided; five all in one direction gives
p = 0.0625. Six of 26 is a 23 percentage point gap.

So Segment B on this manifest can detect only large effects. It is a **protocol
validation and an effect-size direction finder**, not evidence of a small
improvement. If the ladder produces a gap under roughly 20 points — which is the
likely outcome — the correct conclusion is "underpowered, direction observed",
and the next step is the full 500-case LongMemEval-S, not a louder claim.

Designing for this now is cheaper than discovering it after the money is spent.

## Execution order

1. **Free.** Run all 18 credential-free (lane, budget) cells. Inspect coverage
   and every omission `reason`.
2. **Free.** Publish the coverage table. Choose which cells earn a reader call.
3. **Gated.** Segment A Gate 3 — qualify the endpoint with one live call.
4. **Gated.** Segment A Gate 5 — live extraction on one complete case, inspected,
   before authorising the rest of the sample.
5. **Paid.** Extraction over the fixed sample, producing the rung 5 artifact.
6. **Paid.** Reader, then judge, on the chosen cells. Rungs 1-4 first: they are
   the controls, and a broken control invalidates the treatment anyway.
7. **Free.** `compare.py` joins receipts into the final comparison.

Each paid step needs its own fresh approval under the controlled-resource rule in
the project handoff: exact endpoint and model, case window and maximum call
count, prices and hard USD cap, files written, and stop condition.

## Open decisions

These need an owner before step 3, and each changes what the segment measures:

1. **Reader model.** Must have a context window above ~128k tokens if rung 1 is
   to be a genuine full-history control. This is the single most consequential
   choice in the segment.
2. **Judge model.** LongMemEval's official judge, or a pinned substitute? A
   substitute must be justified, because it changes comparability with published
   LongMemEval numbers.
3. **Availability mode.** `released` or `causal`. Both are supported and must be
   identical within any comparison pair.
4. **Budget grid.** Keep all four points, or drop B4 to nearly halve the cost?
5. **Whether to run rung 5 at all on 30 cases**, given the power analysis, or
   move straight to the full sample once the protocol is validated.
