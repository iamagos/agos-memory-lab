# Frozen balanced-30 GPT-5 acquisition and Mem0 context

**Outcome.** The frozen 30-case GPT-5 acquisition completed and passed every
predeclared identity and reproducibility check. It produced 17,201 proposals
and 17,093 active memory records for $11.48821125 in measured model cost,
including qualification. At the constrained 2,400-character lexical operating
point, however, only 8 of 26 answerable contexts contain the exact gold answer
operand. This run measures acquisition and a zero-call retrieval diagnostic; it
does **not** measure reader, judge, or end-to-end accuracy and therefore cannot
be scored against Mem0.

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

## What the 2,400-character diagnostic says

The kernel selected 483 memories across 30 cases: a mean of 16.1 memories,
2,399.1 content characters, and 11.23 distinct sessions per case. All 30
contexts hit the character limit.

For the 26 answerable cases:

| Diagnostic | Result |
| --- | ---: |
| Exact answer operand present | 8 / 26 (30.8%) |
| Exact answer operand absent | 18 / 26 (69.2%) |
| Kernel `recall_any@10` | 88.5% |
| Kernel `recall_all@10` | 84.6% |

Exact-operand survival is uneven by question type:

| Question type | Exact operand present |
| --- | ---: |
| Single-session assistant | 3 / 5 |
| Single-session user | 3 / 4 |
| Multi-session | 1 / 4 |
| Knowledge update | 1 / 4 |
| Single-session preference | 0 / 5 |
| Temporal reasoning | 0 / 4 |

The gap between session-reference recall and exact-operand coverage is the main
finding. Retrieving a memory linked to a gold session does not ensure that the
answer-bearing detail survived extraction and the final character budget. A
reader run at this operating point would conflate reader quality with missing
operands.

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
recall figures may be subtracted from 91.0% or 94.4%.

## Presentation decision

Present the work in four blocks, in this order:

1. **Acquisition completed:** treatment, identities, 30-case coverage, cost,
   and deterministic validation.
2. **Diagnostic finding:** exact operands survive in 8/26 answerable contexts at
   2,400 characters despite much higher session-reference recall.
3. **External context:** quote Mem0's pinned 500-case result only as a published
   reference, with the cohort, retrieval, reader, judge, and metric differences
   adjacent to the number.
4. **Next controlled experiment:** run the same cases through both systems with
   matched retrieval depth/context, reader prompt/model, judge prompt/model, and
   scoring. Report the balanced-30 macro diagnostic separately from the natural
   500-case micro accuracy.

Do not headline an “Agos versus Mem0” percentage from this package. The useful
headline is: **the frozen GPT-5 acquisition is reproducible; operand survival at
the constrained retrieval boundary is now the measured bottleneck.**

## Next gate

Before spending on a reader or judge:

1. freeze development and holdout IDs without inspecting reader output;
2. test retrieval/representation changes with the zero-call operand gate,
   including a budget sweep and at least one non-lexical lane;
3. authorize reader calls only after the operating point materially improves or
   explicitly accept missing-operand cases as part of the end-to-end treatment;
4. add a locally executed Mem0 lane before making a causal or competitive claim.

Any later headline result must keep all non-memory variables fixed and publish
the receipts for both lanes.
