# LongMemEval-S balanced-30 GPT-5 package

This directory is the immutable evidence package for the 2026-08-21 frozen
acquisition described in [`docs/frozen-30-gpt5-run.md`](../../docs/frozen-30-gpt5-run.md).
It covers exactly the cases in
[`manifests/longmemeval-s-balanced-30-v1.json`](../../manifests/longmemeval-s-balanced-30-v1.json).

`runs/` remains ignored scratch and checkpoint space. Files promoted here are
reviewed evidence: do not replace them in place. A different run or treatment
gets a new directory. Every evidence file is below GitHub's 50 MiB warning
threshold, so this package uses ordinary Git rather than Git LFS. JSON and
JSONL files are treated as opaque bytes and their large diffs are disabled in
`.gitattributes`, preserving the recorded hashes across checkout.

## Contents

| File | Bytes | Purpose |
| --- | ---: | --- |
| `azure-gpt-5-text-stratified-30-rank-3-2000-memory.json` | 26,269,857 | Deterministically compiled memory artifact |
| `azure-gpt-5-text-stratified-30-rank-3-extractor.jsonl` | 9,627,629 | Complete frozen extraction: header plus 17,201 proposals |
| `azure-gpt-5-text-stratified-30-rank-3-extractor.state.jsonl` | 4,028,893 | Per-source request, usage, cost, and response checkpoints |
| `azure-gpt-5-text-stratified-30-rank-3-extractor.receipt.json` | 3,887 | Terminal extraction receipt |
| `azure-gpt-5-qualification.json` | 1,585 | Qualification receipt for the served GPT-5 deployment |
| `memory-lexical-2400.json` | 4,577,792 | Zero-call lexical retrieval and governed-selection receipt |
| `memory-lexical-2400-contexts.jsonl` | 131,700 | Gold-free selected contexts bound to the retrieval receipt |
| `memory-lexical-2400-operands.json` | 12,374 | Exact-operand coverage probe bound to those contexts |
| `memory-attribution-audit.json` | 6,438 | High-precision `User`/`Assistant` marker audit |

The extraction receipt names the original scratch filenames. The published
extractor and state files were renamed for a coherent handoff; their contents
are unchanged and their hashes still match the receipt and artifact bindings.

## Verify

On a system with `sha256sum`:

```bash
cd artifacts/longmemeval-s-balanced-30-v1
sha256sum --check SHA256SUMS
```

Validate the artifact contract:

```bash
uv run python -c "from pathlib import Path; import memory; memory.load(Path('artifacts/longmemeval-s-balanced-30-v1/azure-gpt-5-text-stratified-30-rank-3-2000-memory.json'), sha256='010679fb2b9035b5e7d6157bba9e19cf5319a421af0d8dfb2f6b026e6f45a0f6')"
```

Recompile without a model call and compare the resulting bytes:

```bash
uv run python memory.py \
  --dataset s \
  --extractor artifacts/longmemeval-s-balanced-30-v1/azure-gpt-5-text-stratified-30-rank-3-extractor.jsonl \
  --extractor-sha256 995a757c3999a1b47136b29991443106eafece48c48e88a909d5a1a5a2a2ed0c \
  --out runs/frozen30-recompiled-memory.json
```

The recompiled file must hash to
`010679fb2b9035b5e7d6157bba9e19cf5319a421af0d8dfb2f6b026e6f45a0f6`.
