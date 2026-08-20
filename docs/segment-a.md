# Segment A: protocol qualification

Segment A answers whether an experiment is comparable and executable before it
spends enough to estimate memory quality. It does not claim a Mem0 comparison or
a full LongMemEval score.

## Frozen inputs

- Benchmark: LongMemEval revision
  `9e0b455f4ef0e2ab8f2e582289761153549043fc`.
- Dataset: cleaned LongMemEval revision
  `98d7416c24c778c2fee6e6f3006e7a073259d48f`.
- Sample: `manifests/longmemeval-s-balanced-30-v1.json`, five cases per
  question type, selected by a seeded SHA-256 rank.
- Abstention is reported separately. The source corpus has no selected
  abstention candidate in the `single-session-assistant` or
  `single-session-preference` strata, so the manifest records zero rather than
  manufacturing balance.

The same case manifest must be used for raw retrieval and Agos extraction.
Offsets and limits are allowed only after manifest selection and must remain
identical within a comparison pair.

## Gates

1. Verify the corpus and case manifest without credentials.
2. Run `qualify.py --plan` for every endpoint/model/token-field contract.
3. With an approved model and cap, make exactly one qualification call.
4. Run credential-free BM25 on all 30 cases.
5. Run live extraction on one complete case, inspect outputs and actual usage,
   then authorize the remaining fixed sample separately.

An endpoint passes Gate 3 only if the strict schema, forced tool call, returned
model identity, usage fields, and cost bound all pass. “OpenAI-compatible” alone
is not evidence of a pass.

## What follows

Segment B runs the baseline ladder on the frozen 30 cases: full history, BM25,
dense, hybrid, and extracted memories under the same reader and judge. Oracle
sessions remain a ceiling/debug lane.

Segment C is the controlled Qwen scaling experiment. Sweep one model family at
the extractor step first while freezing retrieval, reader, judge, prompts,
token limit, and case manifest. The initial open-weight ladder should use the
same Qwen3.5 release family at 0.8B, 2B, 4B, and 9B; add 27B only if infrastructure
and the first curve justify it. Record exact model revision, serving engine,
quantization, context limit, hardware, and token-limit field. Do not mix a cloud
alias such as `qwen-flash` into that size curve; it is a separate service
baseline. The official Qwen repository lists these release sizes:
<https://github.com/QwenLM/Qwen3.6>.

After choosing an extractor operating point, freeze it and sweep the reader in
a separate experiment. Sweeping extractor and reader together would prevent us
from attributing gains to memory quality versus answer-generation capacity.
