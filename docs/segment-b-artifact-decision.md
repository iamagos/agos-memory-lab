# Segment B artifact decision

**TL;DR.** The received ten-case artifact is valid, but it does not license a
paid reader sweep. At the 2,400-character operating point, four of six
answerable cases are semantically reader-testable and two are not; the artifact
also has no disjoint holdout for E4. Acquire holdout-capable coverage and test a
verbatim-evidence lane before changing the memory record schema.

## Frozen inputs

- Dataset revision:
  `sha256-min-strata-rank-3-v1:167ecba4f48342bc75edb052b4848837032468011cee8ffb8d60aa2d6f4a4843`
- Dataset SHA-256:
  `167ecba4f48342bc75edb052b4848837032468011cee8ffb8d60aa2d6f4a4843`
- Artifact SHA-256:
  `1ccb913a52f02f85b644725559e73a5a014368d4cf003170695a2d63fb20f8a7`
- Extractor revision: `longmem-additive-v3`
- Coverage: 10 cases, 6 answerable and 4 abstention
- Artifact contents: 6,027 proposals, 5,990 active records, 37 rejects, and
  zero replacements

The custom dataset was reconstructed from the pinned 500-case corpus without
changing its identity. For each `(question_type, abstention)` stratum, it takes
the third-lowest `sha256(question_id)`, orders all answerable strata by type,
then all abstention strata by type. The resulting 5,375,901 bytes reproduce the
artifact-bound dataset hash exactly.

## Zero-call results

Memory retrieval used lexical ranking, 100 candidates, no raw episode, and the
same ten cases at every character budget.

| Characters | Selected memories, mean | Truncated cases | Literal operands | `recall_all@10` |
| ---: | ---: | ---: | ---: | ---: |
| 2,400 | 15.0 | 10 | 3 / 6 | 0.83333333 |
| 3,000 | 18.3 | 10 | 3 / 6 | 0.83333333 |
| 6,000 | 36.0 | 10 | 3 / 6 | 0.83333333 |
| 12,000 | 71.0 | 10 | 3 / 6 | 0.83333333 |
| 24,000 | 100.0 | 0 | 3 / 6 | 0.83333333 |

The literal probe is deliberately high precision. Manual inspection promotes
`6071bd76` to reader-testable because the selected memory context contains both
the old ratio, 1 tablespoon per 6 ounces, and the new ratio, 1 tablespoon per
5 ounces. The operating-point semantic partition is therefore:

- reader-testable: `6071bd76`, `70b3e69b`, `ad7109d1`, `bcbe585f`
- not reader-testable: `09ba9854`, `195a1a1b`
- abstention, not applicable to operand coverage: `0ddfec37_abs`,
  `09ba9854_abs`, `29f2956b_abs`, `982b5123_abs`

The two failures are different. `195a1a1b` is a retrieval failure. `09ba9854`
retrieves both gold sessions, but its selected memories omit the taxi price
needed to compute the `$50` saving. Increasing the character budget does not
repair either failure.

The raw-session control is not a drop-in replacement. At 2,400 characters it
has only 1/6 literal operands, versus 3/6 for memories. At 24,000 characters it
has 4/6 literal operands and all six cases are semantically supported, versus
four semantically supported memory cases. Memory extraction is therefore useful
compression at the quota ceiling, but it is not operand-preserving.

## Provenance and update-path findings

A deterministic marker audit found explicit `User` and/or `Assistant`
attribution in 5,899 of 5,990 active records (98.5%). None of the 53 memories
selected across the four 2,400-character abstention contexts was unattributed.
This falsifies the proposed speaker-provenance mechanism on this artifact; a
`role` field is not licensed by E2.

The artifact has zero replacements, as expected from `extract.py` always
emitting an empty `supersedes` list. The answerable knowledge-update context
still contains both old and new ratios, so this one case demonstrates structural
write-path inertness but does not establish that it caused a reader failure.

## Decision

Do not run paid E3 or E4 from this artifact.

1. E3 would have only eight usable development cases: four answerable and four
   abstention.
2. E4 is impossible because the artifact covers no disjoint holdout.
3. A prompt cannot recover the missing taxi operand or the unretrieved
   preference evidence.
4. E2 does not support a role-schema change.

The exact external acquisition request is:

1. Preferred: a frozen 30-case memory artifact made with the unchanged
   `longmem-additive-v3` configuration, its exact dataset file, and its complete
   extractor JSONL.
2. Minimum for proposal-level audit of the current lane: the complete extractor
   JSONL with SHA-256
   `40dd117185497c3ad4a976892b85debbfc3bff56423b9a4254c19b56a7f01643`.

The supplied extractor state is not the complete file: it contains 114 rows for
only three cases and hashes to
`58bd2099b8d3b4aff93ed21bc700b7af725d9ae80105ef644a237fe76c1ab2a9`.

## Next implementation gate

The smallest next representation experiment is a verbatim-evidence lane, not a
new core field:

1. Derive bounded, role-preserving source chunks from the already verified
   sessions.
2. Retrieve and pack those chunks under the same character budgets as memories.
3. Report literal and manually audited semantic operand coverage on byte-frozen
   contexts.
4. Run a local R0 reader only if the verbatim lane improves coverage without
   collapsing the 2,400-character budget.

Only if that experiment wins should a separate schema proposal specify an exact
source span or quote carried by each memory. Such evidence must be validated as
an exact substring of the bound source, remain optional during migration, and
stay out of default rendering until its budget cost is measured.

## Approval boundary

No authenticated or paid call is authorized by this decision. The two pending
DeepSeek qualification outcomes must be resolved before any retry. Any later
DeepSeek reader or Azure judge run needs fresh approval bound to the exact
endpoint, model, case window, request count, current prices, and hard USD cap.
