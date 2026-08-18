# agos-memory-lab

Public, reproducible experiments for
[`agos-memory`](https://github.com/iamagos/agos-memory). The lab consumes the
released package exactly as any external integrator does; it contains no Agos
application code, data, prompts, policies, or adapters.

```text
benchmark case
  -> finite authorized values and retrieval routes
  -> agos-memory decisions
  -> immutable receipt
  -> suite-owned score
```

The kernel decides admission, context retention, bounded selection, and exact
source support. A benchmark script owns dataset acquisition, model calls,
retrieval, storage, and scoring. Those effects do not move into the kernel.

## Run

```bash
uv sync --locked --all-groups
uv run pytest
uv run python smoke.py
```

`smoke.py` is the smallest complete experiment: a source correction, exact
support check, deterministic selection, and content-addressed receipt. CI is
local and free; it performs no networked model calls.

## LongMemEval

`longmem.py` is a direct experiment against the official
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) cleaned
[corpus](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned). It
pins the benchmark code at
`9e0b455f4ef0e2ab8f2e582289761153549043fc` and the dataset at
`98d7416c24c778c2fee6e6f3006e7a073259d48f`.

```text
pinned sessions
  -> raw retrieval                    BM25 or local Qdrant
  -> ranked occurrence routes
  -> agos_memory.select()             order, limits, omissions
  -> bounded dated-session context
  -> content-addressed receipt        IDs and hashes, never corpus text
```

Fetch and verify LongMemEval-S, then run the credential-free BM25 baseline:

```bash
uv run python longmem.py fetch --dataset s
uv run python longmem.py run \
  --dataset s \
  --retriever lexical \
  --contexts runs/lexical-contexts.jsonl
```

The raw BM25 order matches the official session/user implementation, including
its descending-index tie rule. The kernel receives that order as explicit
`SelectionRoute` values. Its default lexical weight is zero so it applies
budgets without silently reranking an upstream retriever; experiments may set
`--lexical-weight` explicitly.

Run dense retrieval, or fuse it with the official BM25 baseline, through an
in-memory Qdrant collection:

```bash
uv run --script longmem.py run \
  --dataset s \
  --retriever qdrant-hybrid \
  --limit 5
```

The script-scoped dependencies keep Qdrant out of the lab and kernel runtime.
Each receipt records the Qdrant and FastEmbed versions, exact model snapshot
revision, model-tree hash, BM25 identity, RRF choice, limits, timings, raw
retrieval metrics, governed metrics, and every kernel outcome. No Qdrant server
or credential is required.

LongMemEval contains duplicate session IDs and a few timestamps later than the
question time. The adapter therefore uses a unique occurrence ID internally,
retains the official session ID for metrics, and treats inclusion in the
released haystack—not its descriptive timestamp—as corpus availability. The
optional contexts file contains the question and exact bounded selection text,
including explicit truncation, but no gold answer.

This command implements retrieval and exact context compilation. It does not
claim an end-to-end LongMemEval QA score: a reader must turn the contexts into
the official `{question_id, hypothesis}` JSONL, and the official evaluator must
judge those hypotheses in a separately authorized model run.

## Add a suite

Use the same narrow pattern:

1. Pin and hash public inputs.
2. Normalize finite values at the boundary.
3. Make time, limits, models, and routes explicit.
4. Emit one immutable receipt separating acquisition, kernel decisions, and
   suite scores.
5. Keep data, credentials, and full outputs in ignored `data/` and `runs/`.

Use PEP 723 inline dependencies when a suite needs its own incompatible stack.
Do not create a common runner, provider registry, storage interface, dashboard,
or adapter framework until two genuinely different suites prove the same
abstraction is necessary.

Paid or authenticated runs are always explicit and remain outside CI.

## Boundary

Public benchmark code may depend on public packages and public or properly
licensed datasets. It must not copy private Agos code or encode product-specific
authority, schemas, prompts, customer data, or evaluation cases.

## License

Copyright 2026 I am Agos, Inc. Licensed under the Apache License, Version 2.0.
