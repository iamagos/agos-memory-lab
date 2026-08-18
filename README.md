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

## Add a suite

Start with one direct, one-word executable such as `longmem.py`:

1. Pin the upstream dataset revision and verify its hash.
2. Normalize it into finite public values at the script boundary.
3. Make time, limits, models, embedders, and retrieval routes explicit.
4. Emit one JSON receipt containing configuration, candidates, decisions,
   latency, tokens, cost, omissions, and suite scores.
5. Keep downloaded data, credentials, and full run output in ignored `data/`
   and `runs/` directories.

Use PEP 723 inline dependencies when a suite needs its own incompatible stack.
Do not create a common runner, provider registry, storage interface, dashboard,
or adapter framework until two genuinely different suites prove the same
abstraction is necessary.

The first target is a cleaned, revision-pinned LongMemEval experiment covering
correction, deletion, restart, partition isolation, and exact source support.
Paid or authenticated runs are always explicit and remain outside CI.

## Boundary

Public benchmark code may depend on public packages and public or properly
licensed datasets. It must not copy private Agos code or encode product-specific
authority, schemas, prompts, customer data, or evaluation cases.

## License

Copyright 2026 I am Agos, Inc. Licensed under the Apache License, Version 2.0.

