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
pinned sessions -----------------------------> raw retrieval
       `-> frozen extraction -> admit -> artifact -> memory retrieval
                 current raw episode fallback -----'
                                                   |
                          BM25 or local Qdrant routes
                                                   v
                   agos_memory.select() -> bounded context
                              memory path -> exact support() reopen
                                                   v
                         direct reader -> official judge
                                                   v
                    run receipts: IDs/hashes, no selected text
```

The raw-session path remains the unchanged control. A memory run may add a
tightly bounded raw-session candidate plane with `--episodes 1`. Official
LongMemEval runs use the complete released haystack for both planes. A stricter
`--availability causal` extraction excludes sessions after the question cutoff,
records them as exact omissions, and applies the same policy to episodic
candidates. Both modes compose through one kernel selection budget. The episode
default is zero, so the memory-only treatment remains unchanged.

The experiment has one baseline ladder. Each step changes one acquisition or
retrieval choice while retaining the same reader and judge:

```text
oracle evidence       debugging ceiling
full raw history      official no-retrieval control
raw BM25              primary credential-free control
raw dense             embedding control
raw hybrid            lexical + dense control
extracted memory      treatment under test
```

Retrieval recall is diagnostic. The end-to-end result is official QA accuracy
alongside context size, calls, tokens, dollars, latency, omissions, and exact
source-support failures. A memory lane succeeds only when it improves the whole
tradeoff against the matching raw lane.

Fetch and verify LongMemEval-S, then run the credential-free BM25 baseline:

```bash
uv run python longmem.py fetch --dataset s
uv run python longmem.py run \
  --dataset s \
  --retriever lexical \
  --contexts runs/lexical-contexts.jsonl
```

Segment A uses a checked-in, source-bound 30-case manifest for cheap protocol
qualification. It contains five cases from each of the six question types and
records the attainable abstention count in every type. Verify it against the
pinned corpus, then pass the same manifest to retrieval and extraction:

```bash
uv run python case_manifest.py verify \
  --dataset s \
  --manifest manifests/longmemeval-s-balanced-30-v1.json

uv run python longmem.py run \
  --dataset s \
  --manifest manifests/longmemeval-s-balanced-30-v1.json \
  --retriever lexical \
  --contexts runs/segment-a-lexical-contexts.jsonl
```

The manifest is applied before `--offset` and `--limit`; its SHA-256 and exact
strata are bound into receipts. It is a qualification sample, not the final
full-corpus score.

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

Add one current episodic candidate to an extracted-memory treatment:

```bash
uv run --script longmem.py run \
  --dataset s \
  --source memories \
  --artifact runs/memory.json \
  --artifact-sha256 SHA256 \
  --retriever qdrant-dense \
  --episodes 1
```

The script-scoped dependencies keep Qdrant out of the lab and kernel runtime.
Each receipt records the Qdrant and FastEmbed versions, exact model snapshot
revision, model-tree hash, BM25 identity, RRF choice, limits, timings, raw
retrieval metrics, governed metrics, and every kernel outcome. No Qdrant server
or credential is required.

Dense indexing uses `--dense-batch` from 1 to 256, defaulting to 32. The resolved
value controls both index chunks and FastEmbed batches and is part of retriever
and run identity. The default follows the frozen equivalence experiment in
[#27](https://github.com/iamagos/agos-memory-lab/issues/27).

Hybrid selection is one explicit graph plan: the fused RRF rank controls order,
while every outcome retains the dense and lexical component ranks that reached
that candidate. Component ranks are attribution evidence, not confidence or
truth.

LongMemEval contains duplicate session IDs and a few timestamps later than the
question time. The default `released` availability treats inclusion in the
released haystack as corpus availability, matching the official benchmark.
`causal` is a separately named Agos diagnostic and must not be reported as an
official LongMemEval score. Both use a unique occurrence ID internally and
retain the official session ID for metrics. The optional contexts file contains
the question and exact bounded selection text, including explicit truncation,
but no gold answer.

`qa.py` owns the experiment, while `model.py` makes one pinned Pydantic AI Chat
Completions request. It checkpoints after each response and resumes only when
the complete request identity still matches. Calls are sequential, have no
retries, reject redirects, and never persist the API key. A durable
`*.pending.json` marker is written before each call; an unknown outcome blocks
another call until an operator inspects and resolves it.

```bash
uv run python qa.py read \
  --contexts runs/lexical-contexts.jsonl \
  --out runs/lexical-hypotheses.jsonl \
  --model READER_MODEL \
  --limit 5 \
  --input-cost INPUT_USD_PER_MILLION \
  --output-cost OUTPUT_USD_PER_MILLION \
  --max-cost HARD_USD_CAP

uv run python qa.py judge \
  --hypotheses runs/lexical-hypotheses.jsonl \
  --dataset s \
  --out runs/lexical-evaluation.jsonl \
  --model JUDGE_MODEL \
  --limit 5 \
  --input-cost INPUT_USD_PER_MILLION \
  --output-cost OUTPUT_USD_PER_MILLION \
  --max-cost HARD_USD_CAP
```

The default OpenAI credential is `OPENAI_API_KEY`. A local compatible server can
use `--provider-id local-vllm --api-key-env LOCAL_LLM_API_KEY --base-url
http://127.0.0.1:8001/v1`. Azure uses the deployment name as `--model` and
defaults to `AZURE_OPENAI_API_KEY`:

```bash
uv run python qa.py read \
  --contexts runs/lexical-contexts.jsonl \
  --out runs/azure-hypotheses.jsonl \
  --provider azure \
  --base-url https://RESOURCE.openai.azure.com/openai/v1 \
  --model DEPLOYMENT
```

Azure OpenAI v1 and Foundry serverless endpoints omit `--api-version`. Legacy
Azure endpoints require it:

```bash
uv run python qa.py read \
  --contexts runs/lexical-contexts.jsonl \
  --out runs/azure-hypotheses.jsonl \
  --provider azure \
  --base-url https://RESOURCE.openai.azure.com \
  --api-version API_VERSION \
  --model DEPLOYMENT
```

Nonzero prices require a hard cap. Before a request, the runner reserves the
UTF-8 prompt byte length, a 256-token chat envelope, and the configured output
limit. Completed provider usage replaces its reservation when deciding whether
the next request fits. Receipts bind provider, endpoint family, API version,
model, optional sampling temperature, and output limit while keeping timeout,
prices, cost cap, and credentials outside request identity. Temperature defaults
to omitted; `null` in the receipt means no temperature was sent. A temperature
that a known model profile would discard fails before network I/O. Both actual
and reserved cost remain visible. Request identity also records the exact
Pydantic AI and OpenAI adapter versions.

## Compatible endpoint qualification

`provider=openai` names the Chat Completions wire adapter. `--provider-id`
separately names the service in every request identity and is required for a
custom endpoint. `--max-tokens-field` freezes whether the adapter sends
`max_completion_tokens` or legacy `max_tokens`.

Before extraction, plan and then explicitly authorize one strict structured
output qualification call. Planning reads no credential, writes no file, and
makes no request:

```bash
uv run python qualify.py --plan \
  --provider-id moonshot \
  --base-url https://api.moonshot.ai/v1 \
  --api-key-env MOONSHOT_API_KEY \
  --model kimi-k2.6

uv run python qualify.py --plan \
  --provider-id deepseek \
  --base-url https://api.deepseek.com/beta \
  --api-key-env DEEPSEEK_API_KEY \
  --model deepseek-v4-flash \
  --max-tokens-field max_tokens

uv run python qualify.py --plan \
  --provider-id alibaba-dashscope \
  --base-url https://dashscope-us.aliyuncs.com/compatible-mode/v1 \
  --api-key-env DASHSCOPE_API_KEY \
  --model qwen-flash
```

The live form additionally requires `--out`, prices, and a positive hard cap
when either price is nonzero. A passing receipt proves this harness observed one
strict tool call, a schema-valid nonce, a served-model identity, and usage. It
does not establish model quality. DeepSeek documents strict tool schemas on its
beta endpoint and the legacy token-limit field; Kimi documents strict tools and
`max_completion_tokens`. These commands have only been planned here, not run
against paid services. See the current
[DeepSeek](https://api-docs.deepseek.com/guides/tool_calls),
[Kimi](https://platform.kimi.ai/docs/api/tool-use), and
[Alibaba Model Studio](https://help.aliyun.com/zh/model-studio/base-url)
documentation before a live qualification.

The judge copies the task-specific and abstention behavior from the pinned
[official evaluator](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py),
including its permissive knowledge-update rule: a response may mention stale
information and still pass if it also contains the update. The official score
is therefore benchmark comparability, not proof of correction safety. The
receipt also reports strict yes/no parse diagnostics without changing the
official label.

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

## Live extraction

`extract.py` freezes one additive, source-only extraction pass. Every request
contains exactly one timestamped source session and never receives the benchmark
question, answer, answer-session IDs, or abstention label. It checkpoints each
completed source and blocks an automatic repeat after an unknown outcome.
Specific information supplied by either the user or assistant is eligible;
generic acknowledgments and assistant restatements are not. The model returns
only bounded memory text, preserving who supplied it. The host deterministically
emits `kind=fact` and `confidence=1.0`, so this treatment does not rank on
model-invented categories or uncalibrated confidence scores.

Plan a window before providing credentials or writing output:

```bash
uv run python extract.py \
  --plan \
  --dataset s \
  --offset 0 \
  --limit 1 \
  --provider azure \
  --base-url https://RESOURCE.openai.azure.com/openai/v1 \
  --model gpt-5 \
  --reasoning-effort minimal \
  --input-cost INPUT_USD_PER_MILLION \
  --output-cost OUTPUT_USD_PER_MILLION \
  --max-cost HARD_USD_CAP
```

The plan performs no model calls, reads no API key, and writes no files. It
reports the exact eligible source count and a conservative ceiling that counts
each UTF-8 prompt byte as one token and reserves the configured maximum output
for every source. `--limit` counts benchmark cases, not model calls.

Only after reviewing that plan should the same frozen request be run:

```bash
uv run python extract.py \
  --dataset s \
  --offset 0 \
  --limit 1 \
  --out runs/gpt5-extractor.jsonl \
  --provider azure \
  --base-url https://RESOURCE.openai.azure.com/openai/v1 \
  --model gpt-5 \
  --reasoning-effort minimal \
  --input-cost INPUT_USD_PER_MILLION \
  --output-cost OUTPUT_USD_PER_MILLION \
  --max-cost HARD_USD_CAP
```

The resulting JSONL is the frozen input to `memory.py`; its adjacent state and
receipt files bind every source, request, response model, usage, cost, latency,
prompt, schema, and output digest. Full-corpus extraction is not a prerequisite
for development: begin with one complete case, inspect the memories and actual
usage, then authorize a larger fixed window only if the evidence warrants it.

## Memory artifact

`memory.py` is the pure, pre-retrieval half of the source-linked LongMemEval
experiment:

```text
pinned sessions + frozen extractor rows
  -> exact source validation
  -> agos_memory.admit()
  -> immutable, content-addressed memory artifact
```

The checked-in extractor is synthetic and exists only to prove the contract
without a model call:

```bash
uv run python memory.py \
  --file tests/fixture.json \
  --sha256 9df07a9961774981e0ed9a0685f02f284063cf8155a3cec18b55c18f0eb67876 \
  --revision fixture-v1 \
  --extractor tests/extractor.jsonl \
  --extractor-sha256 cdd5ddff0f7a1ddd66d4f2a5d8a4a99777000322e4a0b7b1992c6fc50ff026a3 \
  --out runs/fixture-memory.json
```

Every proposal binds one case, session occurrence, date, digest, extractor
identity, and ordinal. The compiler applies proposals chronologically, records
every accept, reject, and replacement, and keeps replacement as lineage. The
pure compiler does not receive benchmark questions, answers, answer-session
IDs, or abstention labels. Retention, retrieval, reader calls, and judging
remain separate stages.

The compiler prints the artifact file SHA-256. Use that exact value to run the
memory path:

```bash
uv run python longmem.py run \
  --file tests/fixture.json \
  --sha256 9df07a9961774981e0ed9a0685f02f284063cf8155a3cec18b55c18f0eb67876 \
  --revision fixture-v1 \
  --source memories \
  --artifact runs/fixture-memory.json \
  --artifact-sha256 ARTIFACT_SHA256 \
  --retriever lexical \
  --contexts runs/fixture-memory-contexts.jsonl
```

The runner validates the artifact hashes, identities, and active records, then
binds them to the exact benchmark and dataset. Retrieval sees only active
admitted memory text. The stateless experiment gives `retain()` the source
time, optional expiry, and zero prior
exposures or attributed uses; it does not claim durable usage history. The
retriever supplies routes, `select()` owns order and bounds, and every selected
record must pass `support()` against its exact pinned session before its text is
rendered. Gold labels enter only the metrics stage.

## Compare

`compare.py` joins already-completed receipts; it never runs retrieval or calls
a model:

```text
session retrieval receipt --\
                           +-> exact contract check -> comparison receipt
memory retrieval receipt --/              |
                                          +-- optional reader receipts
                                          `-- optional judge receipts
```

```bash
uv run python compare.py \
  --sessions runs/session-retrieval.json \
  --memories runs/memory-retrieval.json \
  --sessions-read runs/session-hypotheses.jsonl.receipt.json \
  --memories-read runs/memory-hypotheses.jsonl.receipt.json \
  --sessions-judge runs/session-evaluation.jsonl.receipt.json \
  --memories-judge runs/memory-evaluation.jsonl.receipt.json \
  --out runs/comparison.json
```

Reader and judge pairs are optional, but never one-sided. The comparator
requires the same benchmark, dataset window, retriever, candidate and context
limits, kernel version, reader request, judge request, prices, timeout,
concurrency, and retry policy. Completed runs may use different hard cost caps;
the original signed receipts retain those safety ceilings. The comparator
permits only the intentional retrieval-text difference: raw user turns versus
admitted memory text. Each reader input hash must also match the exact context
artifact named by its retrieval receipt.

The receipt reports raw and governed retrieval metrics, context size, every
retrieval timing, artifact/extractor identity, admission reasons, support
failures, reader and judge QA, calls, tokens, latency, and dollars. A frozen
extractor is an exact input, not a model execution receipt; its file hash is
reported, while live extraction time, tokens, and cost remain owned by the
future bounded extraction stage.

## Boundary

Public benchmark code may depend on public packages and public or properly
licensed datasets. It must not copy private Agos code or encode product-specific
authority, schemas, prompts, customer data, or evaluation cases.

## License

Copyright 2026 I am Agos, Inc. Licensed under the Apache License, Version 2.0.
