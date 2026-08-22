# Frozen balanced-30 GPT-5 acquisition and Mem0 context

**Outcome.** The frozen 30-case GPT-5 acquisition completed and passed every
predeclared identity and reproducibility check. It produced 17,201 proposals
and 17,093 active memory records for $11.48821125 in measured model cost,
including qualification. At the constrained 2,400-character lexical operating
point, 8 of the 12 literal-applicable answerable contexts contain the exact gold
answer string. Complete raw history contains that string for only 12 of 26
answerable cases; the other 14 are derived, normalized, paraphrastic, or
rubric-based. This run measures acquisition and strict zero-call diagnostics;
it does **not** measure reader, judge, or end-to-end accuracy and therefore
cannot be scored against Mem0.

The immutable evidence is stored in
[`artifacts/longmemeval-s-balanced-30-v1/`](../artifacts/longmemeval-s-balanced-30-v1/).

## Frozen treatment

- manifest: 30 cases, five from each LongMemEval question type, including four
  abstention cases; SHA-256 `3806648c2bb691a59bec7df30cc611dcbea34aefdc7e9841b433a4a0ec103a14`
- benchmark: `xiaowu0162/LongMemEval` at
  `9e0b455f4ef0e2ab8f2e582289761153549043fc`
- cleaned dataset: `xiaowu0162/longmemeval-cleaned` at
  `98d7416c24c778c2fee6e6f3006e7a073259d48f`
- extractor: `agos-memory-lab`, revision `longmem-additive-v3`, released
  availability, prompt SHA-256 `5851096a6d8ce493b1327ab71930600d4598a56750d3e08926392029143c60f8`
- model request: Azure `gpt-5`, minimal reasoning, 2,000 maximum output tokens;
  every response was served as `gpt-5-2025-08-07`
- retrieval diagnostic: BM25 lexical, 100 candidates, kernel top-k 100, no raw
  episodes, and a 2,400-character selected-context limit

The balanced manifest is a diagnostic cohort, not the natural 500-case
LongMemEval distribution and not a leaderboard sample.

## Acquisition result

| Measure | Result |
| --- | ---: |
| Cases / source sessions | 30 / 1,436 |
| Source sessions with proposals / empty | 1,429 / 7 |
| Input / output tokens | 3,605,877 / 698,020 |
| Proposals | 17,201 |
| Active records | 17,093 |
| Rejects / replacements | 108 / 0 |
| Extraction model cost | $11.48754625 |
| Qualification model cost | $0.00066500 |
| Total measured model cost | $11.48821125 |
| Extraction wall time | 9,635.945 s (2 h 40 m 35.945 s) |

The extraction finished with complete usage accounting and no missing source
results. `memory.load()` accepted the compiled artifact, its case IDs equal the
manifest exactly, its bound extractor hash matches the complete JSONL, and a
fresh zero-call compile reproduced the memory artifact byte for byte.

The absence of replacements is a substantive structural observation, not a
validation failure: this treatment emitted no supersession edges, so the run
does not exercise update replacement even though it includes knowledge-update
questions.

## Full-history applicability control

The 600,000-character full-history control selected all 1,436 source sessions
with zero truncation and a mean 498,276 selected characters per case. Its exact
gold-string result is 12/26, establishing the maximum population to which the
literal probe applies. Literal absence in the other 14 cases is not evidence
that a reader lacks sufficient evidence.

On those 12 literal-applicable cases, the equal-character lexical diagnostic is:

| Selected characters | Raw sessions | Memories |
| ---: | ---: | ---: |
| 2,400 | 4 / 12 | 8 / 12 |
| 4,800 | 7 / 12 | 9 / 12 |
| 9,600 | 8 / 12 | 9 / 12 |
| 14,400 | 9 / 12 | 9 / 12 |

The complete extracted artifact contains 11/12 source-literal answers. At
14,400 characters, nine are selected, two fall outside lexical top-100, and one
source-literal answer was not preserved by extraction. These are strict string
diagnostics, not end-to-end ceilings.

## What the 2,400-character diagnostic says

The kernel selected 483 memories across 30 cases: a mean of 16.1 memories,
2,399.1 content characters, and 11.23 distinct sessions per case. All 30
contexts hit the character limit.

For the 26 answerable cases:

| Diagnostic | Result |
| --- | ---: |
| Literal-applicable in complete history | 12 / 26 (46.2%) |
| Derived or normalized answer | 14 / 26 (53.8%) |
| Exact string present, applicable subset | 8 / 12 (66.7%) |
| Exact string absent, applicable subset | 4 / 12 (33.3%) |
| Kernel `recall_any@10` | 88.5% |
| Kernel `recall_all@10` | 84.6% |

The uncalibrated 8/26 distribution is retained below only to show where exact
strings occur; it must not be interpreted as answerability by type:

| Question type | Exact operand present |
| --- | ---: |
| Single-session assistant | 3 / 5 |
| Single-session user | 3 / 4 |
| Multi-session | 1 / 4 |
| Knowledge update | 1 / 4 |
| Single-session preference | 0 / 5 |
| Temporal reasoning | 0 / 4 |

The gap between session-reference recall and strict literal coverage remains a
useful extraction/retrieval trace, but it is not the measured end-to-end
bottleneck. Retrieving a memory linked to a gold session does not ensure that an
exact source string survived extraction and selection; conversely, a derived or
normalized answer can be supported without that string. A bounded reader run is
therefore required to determine whether the nonliteral cases contain adequate
evidence.

The marker audit finds explicit whole-word `User` and/or `Assistant`
attribution in 16,874 of 17,093 active records (98.7%). All 66 memories selected
across the four abstention contexts contain at least one such marker. This
high-precision audit makes a broad missing-speaker-marker explanation unlikely
for this artifact, but it is not a semantic role-accuracy evaluation.

## Continuity with the earlier ten-case artifact

The earlier artifact used the same extractor revision, prompt and schema,
requested model, served model version, reasoning effort, output cap, and
availability policy. Only three case IDs overlap, so this is a treatment
stability check rather than a paired result.

| Artifact measure | Earlier 10 cases | Frozen 30 cases | Per-case change |
| --- | ---: | ---: | ---: |
| Proposals | 6,027 | 17,201 | -4.9% |
| Active records | 5,990 | 17,093 | -4.9% |
| Reject rate | 0.614% | 0.628% | +0.014 percentage points |
| Replacements | 0 | 0 | unchanged |

Extraction density and rejection behavior are therefore stable at the larger
coverage. The strict literal-operand diagnostic is 3/6 on the earlier
answerable set and 8/26 here; because the cohorts differ and the earlier result
also received manual semantic review, that change should not be presented as a
paired regression.

## How this relates to Mem0

Mem0's pinned public OSS result for LongMemEval with GPT-5 extraction, GPT-5
answering and judging, and top-k 200 is 455/500, or 91.0%. Its repository also
reports a 472/500, or 94.4%, managed Platform v3 result at top-k 200. Those are
end-to-end accuracy results on all 500 naturally distributed questions. Sources
were read at Mem0's public benchmark commit
[`4b61c5d`](https://github.com/mem0ai/memory-benchmarks/tree/4b61c5d31b9c668a12b4f5e78064248a02c82d2b)
and its
[`longmemeval_gpt5.json`](https://github.com/mem0ai/memory-benchmarks/blob/4b61c5d31b9c668a12b4f5e78064248a02c82d2b/results/oss/longmemeval_gpt5.json)
on 2026-08-21.

| Dimension | This Agos run | Mem0 OSS GPT-5 result | Comparable? |
| --- | --- | --- | --- |
| Cohort | balanced 30 diagnostic | natural 500 | no |
| Retrieval | lexical 100 candidates, 2,400 characters | vector retrieval, top-k 200 | no |
| Reader and judge | not run | GPT-5 / GPT-5 | no |
| Reported measure | acquisition plus operand coverage | end-to-end accuracy | no |
| Model extraction | GPT-5, pinned served version | GPT-5 | partially |

The only defensible comparison today is experimental coverage: Mem0 publishes a
complete end-to-end lane, while this package establishes a reproducible Agos
acquisition lane and exposes its next bottleneck. Neither 8/26 nor the kernel
recall figures is an accuracy measure, and neither may be subtracted from 91.0%
or 94.4%. The 8/26 count is additionally inapplicable to 14 derived or
normalized answers.

## Presentation decision

Present the work in four blocks, in this order:

1. **Acquisition completed:** treatment, identities, 30-case coverage, cost,
   and deterministic validation.
2. **Diagnostic finding:** the exact gold string exists in complete history for
   12/26 answerable cases and survives in 8/12 at 2,400 characters, while
   session-reference recall is substantially higher.
3. **External context:** quote Mem0's pinned 500-case result only as a published
   reference, with the cohort, retrieval, reader, judge, and metric differences
   adjacent to the number.
4. **Next controlled experiment:** run the same cases through both systems with
   matched retrieval depth/context, reader prompt/model, judge prompt/model, and
   scoring. Report the balanced-30 macro diagnostic separately from the natural
   500-case micro accuracy.

Do not headline an “Agos versus Mem0” percentage from this package. The useful
headline is: **the frozen GPT-5 acquisition is reproducible; strict literal
coverage and session recall justify a bounded reader experiment, but do not
predict its answer accuracy.**

## Next gate

Before spending on a reader or judge:

1. freeze development and holdout IDs without inspecting reader output;
2. freeze each candidate ranking, then sweep retrieval depth independently from
   reader character budget; PR #43 owns the ranking-reuse harness seam;
3. report session recall and calibrated literal coverage before authorizing a
   bounded reader experiment over both literal and nonliteral cases;
4. add a locally executed Mem0 lane before making a causal or competitive claim.

Any later headline result must keep all non-memory variables fixed and publish
the receipts for both lanes.
