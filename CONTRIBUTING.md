# Contributing

Make one experiment or harness concern clearly better.

This repository tests the released `agos-memory` package as an external
integrator. It owns public benchmark acquisition, retrieval, model calls,
receipts, and scoring. It does not own kernel behavior or private Agos code,
data, prompts, policies, or adapters.

## Before a pull request

- Read `README.md`, `AGENTS.md`, and the owning script and tests.
- State the one hypothesis, defect, or harness concern being changed.
- Keep protocol changes, refactors, and benchmark-result claims separate.
- Remove unrelated providers, abstractions, formatting, and generated churn.
- Disclose material AI assistance and review every resulting line and claim.

Open a pull request only when the change is coherent and ready to merge.

## Experiments

A treatment changes one named variable against its matching baseline. Freeze the
dataset, benchmark revision, model, prompt, retriever, limits, and relevant
package versions. Preserve immutable receipts that make the comparison
reproducible.

Retrieval recall is diagnostic. Report end-to-end accuracy, context size, calls,
tokens, dollars, latency, omissions, and source-support failures together.
Distinguish official benchmark scores from Agos diagnostics.

Paid or authenticated work is explicit, capped, and outside CI. Never commit
credentials, private data, full provider outputs, `data/`, or `runs/`.

Do not create a common runner, provider registry, storage interface, dashboard,
or adapter framework until two genuinely different suites prove the same
abstraction is necessary.

## Proof

Run the free local gate:

```sh
uv sync --locked --all-groups
uv run pytest
uv run ruff check .
uv run python smoke.py
```

Bug fixes need regression tests. Performance and quality claims need an exact
baseline, treatment, frozen inputs, and receipts. A successful call proves a
protocol only; it does not prove model quality.
