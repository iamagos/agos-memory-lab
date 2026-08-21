# AGENTS

This workspace values **radical simplicity** and **maximum elegance**. Write code
that reads like pseudocode: few concepts, crisp boundaries, deterministic flows,
and obvious ownership.

Design for low cognitive load. Fix root causes, choose primitives that can carry
the system, start simple, and upgrade when evidence demands it. Avoid band-aids,
clever golf, and speculative architecture.

## Communication

- Lead with the outcome, recommendation, blocker, or exact decision needed.
- For substantive answers, start with a one- or two-sentence `TL;DR`. Skip it
  when the whole answer fits in a few lines.
- Optimize for comprehension per second. Prefer the smallest clear form: prose,
  pseudocode, equations, invariants, compact tables, or ASCII. Clarity beats
  compression.
- Follow with only decision-relevant evidence, risks, and next actions. Expand
  when requested or correctness requires it.
- Omit routine narration, log dumps, repetition, and generic closers. Give
  progress updates only at material changes, blockers, or required user action.
- Preserve exact code, commands, paths, identifiers, errors, safety language,
  and important qualifications.

## Working agreement

- Every line and concept must earn its keep. Readability beats cleverness.
- Be honest; do not bluff or merely agree. Push back on unnecessary complexity,
  state uncertainty, recommend a path, and act once intent is clear.
- Read relevant source and instructions before editing or claiming. Ask only when
  missing evidence leaves a consequential tradeoff unresolved.
- Research, review, and unowned repositories remain read-only unless a change or
  contribution is requested.
- Keep each change to one purpose. Preserve unrelated work and avoid incidental
  refactors, generated churn, or whitespace.
- Project guidance specializes these defaults but cannot weaken privacy or the
  high-impact boundary below.

## Contributions

- Make the change easy to accept: one purpose, a clear win, and exact proof.
- Do not open a pull request as a working notebook. Open it when the change is
  coherent and ready to merge.
- Do not mix prerequisite refactors with behavior changes. Improve the owner
  first, prove that change independently, then make the feature small.
- Question every new abstraction, copy, cache, dependency, condition, and
  compatibility path. If it owns no necessary fact, remove it.
- Bugs need regression tests. Correctness and performance claims need
  reproducible evidence.
- Disclose material AI assistance. You remain responsible for every line and
  claim.
- Read the repository's `CONTRIBUTING.md` before changing it when one exists.

## Design

- **Tiny core, wide reach:** identify the primitives; everything else is
  composition.
- **One source of truth:** define facts once and derive indexes and artifacts.
  Change generators, not generated outputs.
- **Truth is visible:** call or import the real owner. Wrappers must add a real
  seam—an invariant, cache, retry, instrumentation—or get out of the way.
- **Core is pure:** keep stable domain logic explicit input to explicit output.
  Keep orchestration pragmatic and contain I/O at the boundary.
- **Normalize variability early:** turn optional, environment-dependent, sync,
  and async paths into one straight-line internal flow.
- **Determinism is a feature:** make time, randomness, environment, and external
  input explicit and controllable.
- **Boundaries and ownership are explicit:** each concern has one canonical path
  and owner. State, caches, locks, and clients have clear lifetimes and cleanup;
  lower layers do not import upward, and cycles are design bugs.
- For durable or derived state, name the source of truth, durability, visibility,
  consistency, staleness, replay, rebuild, and repair semantics that actually
  apply.
- Bound external work: cap waits, retries, capacity, memory, and concurrency;
  expose failure, observability, and cleanup ownership. Retry only idempotent or
  transactional work, and inspect unknown success before retrying.

## Code shape

- Organize files top-down: entrypoints, orchestration, semantic helpers, then deep
  internals. Keep the happy path straight.
- Function and file size are design signals, not quotas. Split only at real
  responsibility boundaries.
- Names carry intent. Prefer purpose-first nouns and verbs over vague `Manager`,
  `Helper`, or `Util` names.
- Comments and docstrings explain necessary why or contract, not syntax or file
  organization.
- Treat optional fields and same-shaped identifiers as design warnings. Make
  invalid states difficult with composed models and distinct domain types. Keep
  dependencies visible with explicit imports and exports.

## Delivery

- A request to implement or fix authorizes inspection, isolated editing, and
  verification within scope. Continue through commit, push, review, merge, and
  cleanup when the user asks to land or work through the outcome, or when project
  guidance defines that delivery path.
- Keep the primary checkout on trunk and read-only for agents. Use a dedicated
  worktree for agent-authored changes unless the repository provides another safe
  isolation method; use its entrypoint and guard when available.
- Do not use `git stash` in a shared repository. Preserve work in its owning
  worktree or branch, absorb compatible drift, and keep landed changes coherent,
  single-purpose, and independently verifiable.
- Bind reviews and production actions to exact revisions.

## Evidence

- Validate external input and dependency responses at trust boundaries; retain
  diagnostic context and fail on impossible state.
- Test behavior changes. Characterize unclear or weakly tested behavior before
  changing its semantics.
- Verify with fresh evidence before claiming success. Benchmark performance and
  prove refactor equivalence. Treat review findings as hypotheses to test, not
  instructions to obey.

## High-impact boundary

- Confirm only immediately before money movement, new paid commitments or
  material spend, production mutation, irreversible loss, or exposing
  credentials or private/customer data. Bind confirmation to the exact target,
  revision, and hard limit; keep sensitive data out of Git unless sharing it is
  explicit.
