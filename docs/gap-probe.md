# Probing the Agos–Mem0 gap

**TL;DR.** This lab has not run a controlled Agos–Mem0 comparison, so it cannot
attribute a gap to the reader or the memory layer. The cheapest discriminating
order is measurement correction, zero-call retrieval and artifact diagnostics,
then frozen-context reader experiments; every hosted call remains a separate,
explicitly approved execution step.

This design supersedes the reader-testability experiment E1 in
`segment-b-pilot.md` with G0.2, and widens its provenance and prompt experiments
into G2 and G8. It does not replace the Segment B ladder.

## Reframe the reported number

The collaborator's ten-case cohort contained four abstention questions. The
benchmark contains 30 abstention questions out of 500. Given the reported
subscores, benchmark-share reweighting is:

```text
0.94 × (5 / 6 answerable) + 0.06 × (2 / 4 abstention) ≈ 0.813
```

That arithmetic changes a reported 70% result into roughly 81% under the natural
answerability mix. It does not produce a benchmark score: ten cases remain too
small, the cohort was not sampled from the benchmark distribution, and the
Mem0 lane was published rather than run here.

LongMemEval-S has the following natural type counts: multi-session 133,
temporal-reasoning 133, knowledge-update 78, single-session-user 70,
single-session-assistant 56, and single-session-preference 30. The frozen
30-case manifest has five of each. Consequently neither pooled accuracy on that
manifest nor its six-type macro average is Mem0's natural-distribution
micro-accuracy. `qa.py` now reports a third, additive
`benchmark_weighted_accuracy`, using exact `(question_type, abstention)` strata
from the hash-verified 500-case reference population. It is `null` when the run
does not observe every population stratum; it never replaces the honest
within-cohort metrics.

Retrieval has the same qualification. `recall_any@k` proves only that at least
one gold session was present. Multi-session and temporal-reasoning comprise 266
of 500 cases and may require every gold session. `recall_all@k` is therefore the
primary retrieval diagnostic. A memory rank is also an item rank, not a session
rank: several selected memories may map to one session. Reports now show mean
selected-item and distinct-session counts beside both recall measures.

## Hypotheses and falsifiers

| ID | Candidate cause | Discriminating test | Cost before live approval | Named falsifier |
| --- | --- | --- | --- | --- |
| G1 | The apparent gap is a cohort artifact. | Reweight the reported cohort, then score all 500 cases. | Free arithmetic; hosted scoring later. | A controlled 500-case gap survives natural-distribution weighting. |
| G2 | Speaker provenance is only prose, so assistant suggestions can read as user facts. | Audit admitted record text and selected abstention contexts for user, assistant, or absent attribution. | Free on an artifact. | Nearly every admitted record preserves attribution in prose and no unattributed assistant-derived record reaches an abstention context. |
| G3 | Retrieval scores contain an answerability signal hidden from the reader. | AUROC for BM25 top-1 score, top-1/top-2 margin, and mean top-k score, answerable vs `_abs`. | Free. | AUROC is approximately 0.5 on the full corpus. |
| G4 | User-only dense indexing blinds assistant-answer cases. | Re-run dense session retrieval with `--dense-full-turns`, fixed otherwise; compare `recall_all@k`, especially single-session-assistant. | Local compute. | Single-session-assistant recall is unchanged. |
| G5 | Embedder capacity is the retrieval bottleneck. | Sweep supported FastEmbed models at fixed corpus, prompt, limits, and full-turn setting. | Local compute and model downloads. | Recall is flat across the ladder. |
| G6 | The live write path never resolves updates. | On knowledge-update cases, count stale and updated records reaching context and their order. | Free on an artifact. | The reader consistently receives and selects the updated fact despite co-existing stale facts. |
| G7 | Extraction loses operands preserved by verbatim evidence. | Equal-character memories, verbatim chunks, and raw-session arms; compare literal coverage, then frozen-reader QA. | Retrieval free; reader local or approved hosted calls. | Memories equal or beat verbatim evidence at equal characters. |
| G8 | The direct reader contract causes hedging and failed abstention. | R0–R3 over byte-identical contexts; report paired movement, answerable/abstention splits, and risk–coverage. | Local first; roughly $1 was reported for a hosted sweep, not verified. | R1–R3 do not beat R0 within a predeclared false-abstention budget. |
| G9 | Reader capability, not memory, explains the movement. | Reader ladder at frozen retrieval and prompt. | Local plus roughly $2 reported hosted cost, not verified. | Accuracy is flat across readers. |
| G10 | Top-10 retrieval capacity is insufficient. | Top-10/50/200 and full-history controls at fixed reader and prompt. | Local retrieval; potentially material hosted input. | Top-50/200 and full history do not improve the paired result. |
| G11 | A development winner generalizes. | R0 and the frozen winner on a seeded, untouched holdout. | One final local/approved hosted pair. | The winner regresses against R0 on holdout. |

The gate is deliberately asymmetric. Material G4/G5 recall movement or
structural G2/G6 confirmation moves artifact and retrieval contracts ahead of
prompt tuning. A prompt winner is not adopted until G11.

## Instruments

### G0.1 — benchmark weighting

The population distribution is read from the verified references used by the
judge. Exact type/abstention strata prevent the 30-case manifest's four
abstentions and balanced types from silently standing in for the natural 500.
The metric is derived reporting data and is excluded from the legacy request
identity projection, preserving run IDs for unchanged completed requests.

### G0.2 — literal operand coverage

`probe.py check-operand` joins a contexts JSONL to verified references. For each
answerable case it records whether the complete case-folded gold answer is a
literal substring of selected context. Abstention cases are `not-applicable`.
This is intentionally a high-precision, low-recall check: paraphrases and
multi-part rubrics can be semantically present while failing the literal test.
Reader experiments may claim literal reader-testability only for the positive
partition.

### G0.3 — mechanical failure coding

`probe.py code-failures` assigns every primary-judge wrong case exactly one
label, in this precedence order:

```text
judge-disagreement
failed-abstention
fact-not-extracted
fact-not-retrieved
lost-in-packing
over-abstention
reader-hedge
reader-unsupported
```

The command uses verified reference operands, source/artifact text, retrieved
identities, selected context, hypotheses, and optional second-judge labels. Its
labels inherit the literal-check limitation; they are deterministic triage, not
a substitute for blinded semantic review.

### G0.4 — recall commensurability

`longmem.py` continues to emit `recall_any@k`, `recall_all@k`, and
`ndcg_any@k`. The selection report additionally emits mean selected items and
mean distinct sessions. These report-only fields are excluded from the legacy
semantic identity projection, so an unchanged retrieval configuration retains
its existing `run_id` and context bytes.

### G3 — answerability

`probe.py answerability` accepts a session/lexical receipt and recomputes exact
BM25 scores from the verified corpus using the same token splitting and official
descending tie order as `longmem.py`. It reports top-1, top-1-minus-top-2, and
mean top-k scores plus overall and per-type AUROC. It does not reinterpret
kernel priority scores as BM25 similarity.

### G4 — full-turn dense control

`--dense-full-turns` changes only dense session embeddings. The hybrid lexical
arm remains user-turn BM25. The flag is rejected outside Qdrant session runs,
is recorded in configuration and retriever identity, and defaults off. Thus the
existing user-only lane remains the permanent control.

## Reader prompt ladder

| Lane | Frozen revision | Contract |
| --- | --- | --- |
| R0 | `longmem-direct-v1` | Official LongMemEval direct prompt, byte-for-byte default. |
| R1 | `longmem-exact-v1` | Preserve step-by-step reasoning; constrain only the final line and requested operand. |
| R2 | `longmem-exact-abstain-v1` | R1 plus `UNANSWERABLE` and user-confirmed provenance rules. |
| R3 | `longmem-judge-then-solve-v1` | Explicit answerability verdict before solving; stop on unanswerable. |

`--reader-prompt` selects the revision. It is bound into prompt text, request ID,
resume validation, and the read receipt. R0 stays the default and must always be
reported. Every ladder run consumes the same contexts file; changing retrieval
and prompt in one contrast is invalid.

Before looking at results, declare the maximum acceptable newly wrong answerable
cases and newly over-abstained cases. Report the per-case 2×2 movement against
R0, not just aggregate accuracy.

## Execution order and stopping rules

1. Verify the frozen manifest and existing lexical receipt.
2. Run G0.2 and G3 on all 500 cases; audit five literal checks against the corpus.
3. Run G4, then G5, at fixed retrieval budgets. These are local, zero-model-call
   retrieval experiments.
4. Audit G2 and G6 only on a source-compatible memory artifact. The received
   artifact covers ten cases from a custom subset, so it cannot be silently
   joined to the pinned full corpus.
5. Run G7 retrieval coverage.
6. Run R0–R3 locally on one frozen contexts file. Keep only variants that satisfy
   the predeclared risk budgets.
7. Qualify the exact DeepSeek endpoint and variant with one separately approved
   call. Then replay surviving contrasts under a hard dollar cap.
8. Use Azure only for the separately approved judge-parity sample.
9. Freeze the winner and run G11 once.

The documented cumulative-limit loop in `README.md` makes one new request per
iteration while preserving a single checkpointed output. It is the sweep
mechanism; no second orchestration layer is justified.

## Endpoint roles and approval boundary

- A local OpenAI-compatible server owns high-call-count prompt and reader
  sweeps. Its served model identity must still be pinned in receipts.
- Requested DeepSeek variant `deepseek-v4-flash` is the candidate hosted reader.
  Official documentation currently lists a 1M context, thinking enabled by
  default, $0.14/M cache-miss input, and $0.28/M output. A 1,024-token
  qualification verified the returned model identity, strict-schema behavior,
  and usage fields. The request did not explicitly bind the effective thinking
  mode, so that fact remains provider-default rather than receipt-measured.
- Azure `gpt-5.6-sol` is judge parity only. Reader and judge identities remain
  separate.

Qualification planning reads no credential and makes no request. Immediately
before any live qualification, DeepSeek reader, or Azure judge call, approval
must bind the endpoint, requested model/returned variant expectation, exact case
window, request count, current input/output prices, and hard maximum USD. No such
call is authorized by this document.

## Provenance ledger

Measured in this repository on 2026-08-21:

- Base revision `10c1755c1743356575bdd3f05cb0347db47720d8`; baseline suite
  `134 passed` before changes.
- The pinned corpus receipt contains 500 cases: 470 answerable and 30
  abstention, with the type counts above.
- Zero-call literal coverage on the existing 500-case lexical contexts:
  233 reader-testable and 237 not-reader-testable answerable cases.
- Zero-call BM25 answerability AUROC on the same receipt: top-1 `0.68042553`,
  margin `0.58929078`, mean top-k score `0.65900709`. This is a probe signal,
  not a threshold or a causal result.
- Received memory artifact SHA-256
  `1ccb913a52f02f85b644725559e73a5a014368d4cf003170695a2d63fb20f8a7`:
  10 custom-subset cases, 6,027 proposals, 5,990 active records, 37 rejects,
  and zero replacements. It is measured evidence for write-path inertness on
  this artifact only.
- One approved DeepSeek request targeted
  `https://api.deepseek.com/beta/chat/completions`, requested
  `deepseek-v4-flash`, allowed 64 output tokens, and had a `$0.001` hard cap.
  It returned HTTP 400 and left pending request ID
  `787d7994c8cce26d72ccd1ff91c99db8d370667e0f5017d5e79aec23116d80d0`;
  no served-model identity, usage, cost, or strict-schema success was measured.
  Local request inspection shows the generic profile forces
  `tool_choice=required`; current DeepSeek V4 compatibility guidance says that
  forcing is unsupported in thinking mode. That is the leading explanation,
  not a measured server error body. The explicit DeepSeek provider profile now
  changes the field to `auto`, with regression coverage.
- One approved retry used that DeepSeek profile with the same endpoint, model,
  64-token limit, and `$0.001` cap. It passed HTTP request validation but ended
  as `chat_output_limit_exceeded`, leaving pending request ID
  `79f6bace6f2599e99d58b7f3a721b6bfc9998a0cffefb28575fc2709b2d0d90d`.
  It produced no success receipt or usage measurement. This is evidence that
  64 output tokens are insufficient for the default thinking path, not evidence
  that strict structured output succeeds.
- One approved 1,024-token qualification then succeeded. Request ID
  `7186b5aa7ceffc5f54ce3082a7e640f6a6d3a99ac2fc4f47812654313203d63e`
  returned model `deepseek-v4-flash`, validated the strict nonce schema, used
  392 input and 92 output tokens, and measured `$0.00008064` at 2.401464
  seconds. Its run ID is
  `f652afa0f652c76460498efdbf36068e4c3ed9af32302b1e2ce335697476f0f6`
  and receipt SHA-256 field is
  `0fc474797d158dc09a4be93a61791e753a1d6ef52d112ed7901f5a91fde56098`.

Inferred or externally reported, not verified here:

- The collaborator's 70% Agos and 100% Mem0 ten-case figures, their exact lane,
  and the claim that all three losses were reader-only.
- Mem0's published 91.0%, its reported GPT-5 reader, top-200 configuration, and
  per-type figures. No controlled Mem0 lane exists in this tree.
- DeepSeek's official pages currently report V4 Flash as 284B total / 13B
  active, a 1M context, dual thinking modes, and the prices above. Those are
  provider claims, not endpoint observations, and prices must be re-read at
  approval time.
- Literature results motivating provenance records, verbatim controls,
  answerability gating, reader ladders, and judge divergence. They motivate
  tests but do not decide them.

## Explicit non-goals

- No claim that Agos is stronger or weaker than Mem0 without a controlled Mem0
  lane using matched reader, judge, prompt, retrieval budget, and cohort.
- No `role` field on `Record`. If G2 warrants schema ownership, write a separate
  artifact-contract proposal covering compatibility and migration.
- No HAD implementation. HAD is undefined in this change and remains a separate
  later pathway.
- No paid or authenticated model call, extraction, production mutation, or new
  credential handling.

## Reported sources

- [Mem0 memory benchmarks](https://github.com/mem0ai/memory-benchmarks#results)
  and its [reported OSS LongMemEval result](https://github.com/mem0ai/memory-benchmarks/blob/main/results/oss/longmemeval_gpt5.json)
- [LongMemEval](https://arxiv.org/abs/2410.10813) and its
  [repository](https://github.com/xiaowu0162/LongMemEval)
- [Two Axes of LLM Abstention](https://arxiv.org/abs/2607.08456)
- [Bridging the Detection-to-Abstention Gap](https://arxiv.org/html/2605.28070v1)
- [Eywa](https://arxiv.org/pdf/2605.30771)
- [Verbatim Chunks Beat Extracted Artifacts](https://arxiv.org/pdf/2601.00821)
- [SmartSearch](https://arxiv.org/pdf/2603.15599)
- [Chain-of-Memory](https://arxiv.org/pdf/2601.14287)
- [Let Me Speak Freely](https://arxiv.org/html/2408.02442v1)
- [DeepSeek models and current pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [DeepSeek strict tool-call mode](https://api-docs.deepseek.com/guides/tool_calls)

The literature citations are carried from the approved plan and were not
independently audited for this implementation. The two DeepSeek pages were
checked on 2026-08-21; endpoint behavior still requires qualification.
