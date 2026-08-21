# Frozen 30-case artifact acquisition

**TL;DR.** Request one memory artifact and its complete extractor JSONL for the
already frozen balanced manifest. Do not request reader outputs, judgements, or
a new dataset. The lab already owns and verifies every other input.

## Exact request

Please produce these two opaque files:

1. A `memory.py` artifact covering exactly
   `manifests/longmemeval-s-balanced-30-v1.json`.
2. The complete extractor JSONL used to compile that artifact, including its
   header and every proposal row for all 30 cases.

Use these identities unchanged:

- benchmark: `xiaowu0162/LongMemEval` at
  `9e0b455f4ef0e2ab8f2e582289761153549043fc`
- cleaned dataset: `xiaowu0162/longmemeval-cleaned` at
  `98d7416c24c778c2fee6e6f3006e7a073259d48f`
- dataset SHA-256:
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- manifest SHA-256:
  `3806648c2bb691a59bec7df30cc611dcbea34aefdc7e9841b433a4a0ec103a14`
- extractor: `agos-memory-lab`, revision `longmem-additive-v3`, schema
  `agos-memory-lab-extractor-v1`
- current comparison configuration: released availability, Azure `gpt-5`,
  reasoning effort `minimal`, and 2,000 maximum output tokens

Return the SHA-256 and byte size of each file separately. Transfer both as
opaque bytes; do not paste them through a newline-normalizing channel.

## Not requested

- no hypotheses, contexts, reader answers, evaluations, or judge output
- no reproduction of the collaborator's reported 70% result
- no custom ten-case dataset
- no role-field or supersession schema change
- no extension of the extractor configuration without a separate proposal

## Acceptance checks

The package is accepted only when all of these hold:

1. The transferred file hashes match the sender's hashes.
2. `memory.load()` validates the artifact against its supplied SHA-256.
3. The artifact binds the benchmark, dataset, and extractor identities above.
4. Artifact case IDs equal the 30 manifest IDs exactly, with no extras or
   omissions.
5. The complete extractor JSONL hashes to the extractor input SHA bound inside
   the artifact.
6. Every extractor source ID, source date, and source digest reopens against the
   pinned corpus.
7. Recompiling with `memory.py` reproduces the artifact byte-for-byte.
8. The ten previously inspected custom cases are not treated as a holdout. The
   balanced manifest receives a deterministic development/holdout split before
   any reader result is inspected.

Failure of an identity check is an intake failure, not permission to weaken the
hash contract. Preserve the received bytes and request a corrected package.

## Work allocation after arrival

- `codex/segment-b-e0-e4` owns opaque intake, validation, and split manifests.
- `codex/verbatim-evidence-probe` remains a closed negative control.
- A new reader branch starts only after the accepted artifact yields a disjoint
  E4 holdout and the 2,400-character operand gate is rerun.
- DeepSeek and Azure calls remain separately approved execution steps.
