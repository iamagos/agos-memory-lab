# Segment B pre-flight: the reader contract pilot

Segment B holds the reader, judge, prompts, and budget **fixed** across every
rung of the ladder. That is what makes the ladder a measurement of acquisition
and retrieval rather than of answer generation.

The collaborator finding changes what that fixed reader has to be. It reports
that on ten questions our retrieval already contained the necessary evidence in
every failed case, and that the three losses were reader behaviour: an overbroad
numeric answer, and two failures to abstain. If that is right, then freezing the
current reader and running the full grid would spend the entire Segment B budget
measuring a defect that is constant across all five rungs, and would depress
every lane by the same amount.

So this document specifies the small experiments that must settle the reader
contract **before** Segment B is committed. They are deliberately sized at
roughly one percent of Segment B's token volume.

Read `docs/segment-b.md` first. This pilot does not modify it; it supplies the
answer to its open decision 1 and adds one it does not yet have — which reader
*prompt* the ladder freezes.

## What the finding licenses, and what it does not

**Licenses.** Treating exact answer construction, abstention, and evidence
provenance as the first thing to fix. Treating the ten questions as development
cases. Not buying top-200 retrieval or more infrastructure yet.

**Does not license.** Any claim against mem0. The reported comparison is not
controlled: different reader, different retrieval budget, different prompt, and
a mem0 number that is a published result rather than a lane we ran. Under the
handoff's mem0 rule that is a *reported* figure standing next to ours, and it
must be labelled that way in any writeup.

It also does not license quoting 70% against 100%. Ten questions with three
discordant cases is p = 0.25 under McNemar — weaker even than the 26-case power
problem already documented in `docs/segment-b.md`. These ten can diagnose. They
cannot score.

## Three things to settle before any call is made

### 1. The finding is not reproducible here, and we are not going to reproduce it

`runs/` here contains no hypotheses file, no evaluation file, and no memory-lane
retrieval receipt. `runs/HANDOFF.md` states that no paid or authenticated request
has been made. The reported ten-question result came from somewhere else, and
nothing in this tree pins which ten questions they were, which lane produced
them, or at what budget.

The retrieval budget in the finding — "Top 10 + 1 episode" — and the reported
~2,888 input tokens per query are consistent with `--source memories --top-k 10
--episodes 1`, since ten extracted memories plus one raw episode is roughly one
session of text. That is an inference, not a receipt.

**Reproducing it is not worth the effort, because that configuration cannot run
here.** At ~3,900 metered tokens it is roughly 4x the per-call ceiling, so
retrieval and contexts must be regenerated at a smaller budget whatever we
receive. Once they are regenerated, the reader baseline is ours to measure and
the collaborator's hypotheses describe a lane we are not using.

So the 70% keeps the status this document already gave it: motivation for
prioritising the reader, not a number the pilot is chasing or defending. What we
need from the collaborator collapses to the one artifact we cannot manufacture,
which is what E0 now asks for.

### 2. Changing the reader prompt changes comparability

`qa.py` uses `_PROMPT_REVISION = "longmem-direct-v1"`, which is the official
LongMemEval direct reader prompt. It is the reason our numbers are comparable to
published LongMemEval results at all.

"Fix the reader contract" therefore cannot mean "edit the prompt." It has to mean
"add pinned prompt revisions as an explicit, swept variable, and keep the
official one as a permanent control lane." The receipt machinery already supports
this — `prompt_revision` is bound into request identity and written to every
receipt — but `qa.py read` has no flag to select one. That is the only code
change this pilot requires.

Rule for the rest of the segment: **the official-prompt lane is always run and
always reported.** A non-official prompt may be reported beside it, never
instead of it.

### 3. The quota is a fixed design constraint, not a scheduling problem

Quota on `gpt-5.6-sol` is **1,000 TPM / 1 RPM and cannot be raised.** That is not
a delay to plan around; it is a constraint the experiment has to be designed
inside, and it eliminates most of Segment B outright.

Two things follow from how the meter works. Azure bills a request against the
window as prompt tokens **plus the requested `max_tokens`**, and a request whose
estimate exceeds the whole per-minute allowance is refused rather than queued. So
the ceiling applies per call, and the output budget is part of it.

Sizes below are **measured**, not estimated — tokenised with `o200k_base` over
the pinned corpus and the actual `qa.py` prompt builders, on the frozen 30-case
manifest. Provenance is at the end of this document.

| Call | Prompt tokens | `--max-tokens` | Metered | Verdict |
| --- | ---: | ---: | ---: | --- |
| Gate 3 qualification | small | small | small | fits |
| Judge, per case (max of 30) | **488** | 10 (current default) | **498** | **fits, 2x headroom** |
| Reader, memory-only at `--chars 2400` | **542** | 300 | **969** | **fits, ~3% margin** |
| Reader, memory lane + 1 episode (the reported config) | ~2,900 | 1,000 | ~3,900 | ~4x over |
| Reader, one raw session | ~3,100 | 1,000 | ~4,100 | ~4x over |
| Reader, raw BM25 at top-10 | ~31,000 | 1,000 | ~32,000 | ~32x over |
| Extraction, per source session | ~3,100 | — | ~3,600 | ~4x over |

RPM is not the binding constraint. The whole pilot is 140 calls, which at 1 RPM
is about two and a half hours of wall clock — acceptable for a pilot. **Only
per-call size binds.**

**What survives on this endpoint.** Exactly one reader lane, and the operating
point is now exact rather than approximate:

```text
--source memories --episodes 0 --chars 2400 --max-tokens 300
```

The reader prompt's fixed overhead is 127 tokens on the manifest's longest
question, leaving 573 tokens of context at `--max-tokens 300`. Real LongMemEval
text tokenises at 4.43 to 5.19 characters per token, so `--chars 2400` is at most
542 tokens: **969 metered worst case against a 1,000 ceiling.** The margin is
thin but real, and it is thin in the safe direction — the 4.43 floor is the
worst ratio observed across all 30 cases, not the mean.

The judge needs no change at all. Its measured worst case is 498 metered tokens,
which is the one genuinely comfortable number in this document.

**What is permanently off this endpoint.** Every raw-session lane, and therefore
Segment B rungs 0 through 4 — a single session does not fit, so no amount of
budget tuning helps. The full-history control. And live extraction, which is the
one that hurts: at ~48 sessions per case, one case is ~150,000 tokens and no
individual session call is admissible anyway.

**The consequence that reorders the whole plan.** The only reader lane that fits
is the memory lane, and the memory lane requires an artifact that this endpoint
cannot produce. So recovering the collaborator's existing memory artifact (E0) is
no longer merely good hygiene — it is the difference between having a treatment
lane and not having one. Failing that, the extractor moves to a local
OpenAI-compatible server, which `extract.py` already supports via `--base-url`
and which is Segment C's plan arriving early.

### 4. Two live-call hazards that this quota makes routine

Neither is speculative; both are visible in the code.

**A 429 halts the run and needs a human.** `model.py` builds its agent with
`retries=0` and `UsageLimits(request_limit=1)`, and maps a rate-limit response to
`chat_http_error:429`. `qa.py` writes a durable pending marker *before* each call
and clears it only after the record is written, so a refused request leaves a
marker that `_recover_pending` refuses to resolve automatically. At 1 RPM, a
naive `qa.py read --limit 10` issues its second call within seconds, is refused,
and stops with an unknown outcome to resolve by hand.

So pacing is mandatory infrastructure. Either add a `--min-interval` to `qa.py
read`/`judge`, or drive it externally at one call per minute with `--offset N
--limit 1`, which the existing checkpoint-and-resume design already supports. The
external loop needs no code change and is the honest first choice.

**The judge may return nothing.** `gpt-5.6-sol` is a reasoning model, and
reasoning tokens are drawn from the completion budget. The judge's default
`--max-tokens 10` is sized for a bare `yes`/`no` and may be entirely consumed by
reasoning, in which case `model.complete()` raises `chat_response_invalid` on an
empty string — a failure that has nothing to do with quota and would look like a
broken judge. `--reasoning-effort none` or `minimal` is the lever, and `model.py`
already rejects a temperature that a reasoning profile would discard.

Both questions are answered by the one call in Gate 3.

## The experiments

Ordered free first, exactly as Segment B's execution order is. Each names what
would falsify the finding, because an experiment that cannot come out the other
way is not worth running.

### E0 — Acquire the one input we cannot manufacture (free)

**Question.** Do we have a memory artifact, and does it cover enough cases to
finish the pilot?

This experiment used to be about reproducing the collaborator's ten-question run.
It no longer is, and dropping that goal makes the pilot both cheaper and cleaner.
Two facts collapse it:

- **His retrieval configuration cannot run here anyway.** Top-10 plus one episode
  is ~4x over the per-call ceiling, so retrieval and contexts get regenerated at
  `--chars 2400 --episodes 0` regardless of what he sends.
- **Once we generate our own contexts, we run our own R0 baseline**, and that
  baseline *is* E3's control. His hypotheses and judgements would be a second,
  non-comparable measurement of a configuration we are not using.

So the reported 70% reverts to what it always was by this document's own
standard: motivation, not a number we are chasing. Nothing downstream depends on
verifying it.

**Procedure.** Ask for one file, and one fact.

1. **The memory artifact** from `memory.py`. This is the whole ask. Extraction
   runs ~4x over the per-call ceiling, so this endpoint cannot produce a single
   memory at any budget — the artifact is the one input the quota permanently
   blocks us from making ourselves, and without it there is no treatment lane.
2. **How many cases the extraction covers, and whether he can extend it.** See
   the coverage constraint below; this is now the question that decides whether
   the pilot can finish.

Useful but not blocking: the **extractor JSONL**, which allows the artifact to be
recompiled and audited at the proposal level rather than the record level. The
artifact already binds `extractor_name`, `extractor_revision`,
`extractor_config_json`, and the extractor SHA-256 internally, so its provenance
is reportable without the file.

Not needed: the retrieval receipt, contexts JSONL, hypotheses, or evaluation.

The artifact's own SHA-256 is worth having as a transfer check, but it is not a
dependency — `longmem.py` takes `--artifact-sha256` as a value we supply, so it
can be computed on arrival.

**The coverage constraint, which is the real risk.** A memory artifact only
contains records for the cases that were extracted. If his extraction covered
only the ten cases he tested, then E3 can run and **E4 cannot** — the 20-case
holdout has no memories, and we cannot extract them. That would leave the pilot
able to fit a reader contract but not to confirm it, which is the one outcome
worth avoiding.

So the extension question goes out with the initial ask, not after E3 discovers
the gap. Coverage itself is readable directly off the artifact once it lands, by
counting distinct `case_id` values across its records — no need to ask twice.

**Case selection follows the artifact, not the other way around.** Whatever the
artifact covers is the population. Split it into a development set and a disjoint
holdout, freeze both with `case_manifest.py`, and record their SHA-256 values
here. If he identifies which cases he already inspected, put those in the
development half; if he does not, a seeded split is fine, because we are no
longer inheriting a result fitted to them.

**Precondition: line endings silently break every hash. Resolved 2026-08-20.**

This was not hypothetical — it was failing in this checkout. `core.autocrlf` is
`true` here, upstream main carried no `.gitattributes`, and so `tests/fixture.json`
was checked out with 74 CRLF pairs. Its SHA-256 was then `2f796929…` rather than
the pinned `9df07a99…`, and `longmem.py` correctly refused with
`dataset_sha256_mismatch`. Two tests failed for exactly this reason.

PR #22 fixes it from both directions, and it has been cherry-picked onto this
branch as `10c1755`: it adds a `.gitattributes` pinning `*.json`, `*.jsonl`,
`*.md`, `*.py`, `*.toml`, and `*.yml` to `eol=lf`, and it replaces the brittle
literal `run_id` assertion with an internal-consistency check. Because attributes
do not renormalise files already on disk, the 16 affected working-tree files were
re-checked out; `tests/fixture.json` now hashes to the pinned `9df07a99…` and the
suite is **134 passed**, matching the count `runs/HANDOFF.md` recorded for
`ef35b97`. Five files keep CRLF — `.gitignore`, `.python-version`, `LICENSE`,
`NOTICE`, `uv.lock` — because they match no explicit pattern and none is
hash-pinned.

The consequence for E0 stands, and now applies to a single irreplaceable file.
The memory artifact is hash-verified before use, and any transfer that normalises
newlines invalidates it. Move it as opaque bytes — archived, not pasted and not
routed through anything newline-aware — and hash it on arrival rather than at
first use. A mismatch caught on arrival is a transfer to repeat; the same
mismatch caught mid-pilot looks like a reproducibility failure in the lab.

**Exit criterion.** The artifact validates against `memory.py`'s contract, its
SHA-256 is recorded, its case coverage is counted, its `extractor_revision` is
checked, and the development and holdout manifests are frozen with their hashes
written down.

**The revision check, and why it is not a formality.** `extract.py` writes a
header row carrying `name`, `revision`, and `config`; `memory.py` binds those
into the artifact as `extractor_name`, `extractor_revision`, and
`extractor_config_json`. So the artifact states which extraction prompt produced
it, readable without the extractor JSONL and without running `extract.py` at all.

It must read `longmem-additive-v3`, which is `_PROMPT_REVISION` at
`extract.py:23` in this tree. If it names an earlier revision, then E2's whole
mechanism — that `longmem-additive-v3` deliberately harvests assistant
recommendations "even when they are not personal facts about the user" — is an
analysis of a prompt that did not produce these memories, and E2's finding would
not transfer to the artifact under test. A mismatch does not invalidate the
artifact; it means E2 must be re-derived against the revision actually recorded,
and that any conclusion about extraction is scoped to that revision.

Record the value either way. It is part of what the pilot can claim, since the
extraction stage is one we are inheriting rather than observing.

**Failure mode.** If no artifact can be produced, there is no memory lane, and
therefore no reader lane that fits this quota at all. The pilot then moves
wholesale to a local extractor and reader, which is Segment C's infrastructure
arriving early and is the fallback already named in the execution order.

**Standing consequence.** Any cases the collaborator already inspected are
development cases. They may be iterated on and fitted to; they may never appear
in a headline number, and E4's holdout must be disjoint from them.

### E1 — Is the evidence in *our* context? (free)

**Question.** For each case in the artifact, does the context our own memory lane
selects contain the literal operand the gold answer requires?

The collaborator's version of this question was whether his three failures were
reader defects or retrieval defects. Since we are no longer reproducing his run,
the question becomes the one that actually governs the pilot: **whichever cases
fail E1 cannot be used to test a reader contract at all**, because no prompt can
recover an operand that is not in the context.

The distinction that makes this worth checking is unchanged. Recall metrics score
the *answer session*; the reader consumes the *selected context*. On the memory
lane those are not the same object — the context holds extracted memory text, and
a memory derived from an answer session can drop the operand the question needs
while the session still counts as recalled.

**Procedure.** Run the memory lane at the pilot's operating point, then for each
case read the exact context from the contexts JSONL and search for the literal
operand its gold answer requires — for an exact-value case, whether `$50` or the
figures that determine it are present, rather than only figures that would
support a range. No model call. The gold answers are short: median 11
characters, which makes this a mechanical string check on most cases.

**Output.** A partition of the artifact's cases into *reader-testable* (operand
present) and *not reader-testable* (operand absent). E3 and E4 draw only from the
first group; the second is evidence about extraction and selection, and feeds E2.

**Falsifier for the whole pilot.** If the operand is absent on most cases, the
reader contract is not the first defect and E3 is the wrong experiment — the
pilot pivots to E2 and the artifact contract as its primary line.

### E2 — Audit provenance in the memory lane (free)

**Question.** Can the reader tell a user-stated fact from an assistant
suggestion?

There is a concrete mechanism for the "generic assistant recommendations treated
as user-specific evidence" failure, and it is structural rather than a matter of
reader diligence:

- Raw sessions render with explicit speaker prefixes — `_session_text()` in
  `longmem.py:1252` emits `user:` / `assistant:` on every turn.
- Memory records render as bare `record.text`. `memory.Record` has fields for
  `kind`, `text`, `confidence`, and source lineage, but **no role or speaker
  field**, and `kind` is deterministically `fact` for every record.
- The extraction prompt `longmem-additive-v3` explicitly harvests assistant
  content — "extract specific recommendations, instructions, solutions,
  researched facts, plans, and commitments, even when they are not personal facts
  about the user" — and asks only that the model "preserve who supplied the
  information" in prose.

So on the memory lane, attribution survives only if the extractor happened to
write it into the sentence. If it did not, an assistant's suggestion arrives at
the reader as an undifferentiated `fact`, and a reader that then answers a
question about the user is behaving reasonably given its input.

**Procedure.** Over every case the artifact covers, classify each admitted
record's text as user-attributed, assistant-attributed, or unattributed. Then, on
the abstention cases specifically, check whether an unattributed
assistant-derived record reaches the selected context. This runs on the artifact
alone and needs nothing from the collaborator beyond the artifact itself.

**Falsifier.** If attribution is preserved in the text on essentially every
record, this mechanism is not the cause and E3's provenance clause is the only
remaining lever.

**If it is confirmed**, the fix is not a prompt at all. It is a structural role
field on the record, carried through admission and rendered into the context —
which is a change to the artifact contract and belongs in its own scoped
proposal, not smuggled into a prompt experiment.

### E3 — Reader prompt A/B at fixed context (paid, small)

**Question.** Does an explicit answer-and-abstention contract score better than
the official prompt on identical contexts, and which cases does it move in each
direction?

R0 is not a formality here. Since we are no longer inheriting the collaborator's
70%, **R0 is the baseline** — the first measurement of this lane at this budget,
and the only thing R1 and R2 are compared against.

**Design.** Hold the context byte-identical — replay the exact contexts artifact
that E1 and E5 produced — and vary only the prompt revision. That isolates the
reader completely: no retrieval runs, nothing else can move.

| Variant | Revision | Change from control |
| --- | --- | --- |
| R0 | `longmem-direct-v1` | Control. Official prompt, unmodified. |
| R1 | `longmem-exact-v1` | R0 plus a final-line answer contract: the exact value, no range, no hedge, when the context determines it. |
| R2 | `longmem-exact-abstain-v1` | R1 plus an abstention clause and a provenance rule — only statements the user made count as evidence about the user; absence of a fact is not evidence of it. |

**Volume.** 10 reader-testable cases x 3 variants = 30 reader calls at ~700 input
tokens and `--max-tokens 300`, plus 30 judge calls at ~510 metered. About 37,000
tokens total, and 60 minutes of wall clock at 1 RPM. The case count is whatever
E1 declares reader-testable, capped at 10 to keep the run inside an hour.

**Code change required.** `qa.py read` gains `--reader-prompt`, selecting from a
frozen registry of revisions. The revision string is already part of request
identity and already lands in the receipt, so no receipt-contract change is
needed. R0 stays the default. Pacing is handled by the external one-call-per-
minute driver, not by a new flag, unless E3 proves the loop unwieldy.

**The quota changes what R1 is testing.** At 2,400 characters the context is
roughly a fifth of what the collaborator's configuration held, and the raw
episode is gone entirely. This is a leaner lane than anyone has measured, so R0's
score is a new fact rather than a confirmation. E1 and E5 together decide whether
the lane is diagnosable at all at this size; if it is not, E3 moves to a local
reader with the rest.

**Measurement.** Official judge accuracy on the 6 answerable cases and abstention
accuracy on the 4, reported separately exactly as Segment B does. Plus the strict
yes/no parse diagnostics the judge already emits.

**How much R1 is worth, measured.** The reported failure — answering a range
where the gold answer is a single figure — is not an isolated case. Across the
pinned 500-case corpus, **159 cases (32%) have a short exact-value gold answer**
such as `$800`, `16GB`, `4 hours`, or `7 pm`, and 32 of those are stored as bare
numbers rather than strings. They concentrate in `multi-session` (92 of 133),
which is the largest stratum in the benchmark and the one whose answers require
aggregating across sessions — exactly where a hedged range is the tempting
output.

The official judge's rule for these types is that a response containing only a
subset of the required information scores no. So an answer contract that forces a
single exact value where the context determines one is aimed at roughly a third
of the benchmark, not at one anecdote. That is the strongest available argument
for settling the reader before spending on the ladder.

**Falsifier.** If R1 and R2 do not beat R0, the defect is not the prompt, and the
reader-model choice — Segment B open decision 1 — becomes the variable to sweep
instead.

**Regression check that matters more than the win.** An abstention clause that
buys 2 abstentions by losing 2 answerable cases is a wash disguised as progress.
Report the 2x2 movement per case, not the aggregate.

### E4 — Holdout confirmation (paid, small)

**Question.** Does the winning contract survive on cases it was not tuned on?

E3 selects among three variants on ten cases whose failures we have already
inspected. Its result is not evidence of anything until it transfers.

**Procedure.** Draw up to 20 further cases stratified across the six question
types, seeded and frozen as `manifests/reader-holdout-20-v1.json`, **disjoint
from E3's cases** and drawn only from those E1 declared reader-testable. Run the
E3 winner and R0 only — two lanes, not three, because the loser has already been
eliminated.

**This is what E0's coverage question decides.** The holdout exists only if the
artifact covers enough cases to supply it. If it does not, E4 cannot run here at
any budget, and the pilot ends at "direction observed on the development set" —
which is precisely the underpowered conclusion `docs/segment-b.md` warns against
treating as a result.

**Volume.** 20 cases x 2 variants = 40 reader + 40 judge calls, roughly 50,000
metered tokens and 80 minutes at 1 RPM.

**Exit criterion.** The winner is adopted as Segment B's frozen reader contract
only if it is at least neutral against R0 on the holdout. A win on dev and a loss
on holdout means the dev set was fit, and R0 stays frozen.

This is the only number in the pilot that may be quoted, and it should still be
quoted as a direction, not a score.

### E5 — Does the evidence survive at the quota ceiling? (free)

**Question.** At `--chars 2400 --episodes 0` — the largest memory-lane context
this endpoint admits — how many of the artifact's cases still contain the
evidence their answers need?

The quota promoted this from a cost optimisation to a **gate on whether E3 can
run here at all.** If evidence collapses at 2,400 characters, any accuracy
measured at that budget is a budget artifact and says nothing about the reader
contract.

E5 and E1 are two views of one sweep: E5 finds the budget, E1 partitions the
cases at it. Run them together.

**Procedure.** Sweep `--chars` downward with `--top-k 100` and `--candidates 100`,
exactly as the Gate 4 follow-up did, and record answer-evidence coverage at each
point down to 2,400. Every omission's `reason` field is inspected; anything other
than `char_budget` is a bug, not a result. Then apply E1's literal-operand check
at 2,400 specifically.

Also verify the tokenisation assumption directly: extracted memory text is denser
than session dialogue and may tokenise worse than the 4.43 characters-per-token
floor measured on raw sessions. If it does, `--chars` comes down until the
measured worst case clears 1,000.

**Decision it produces.** Coverage holds at 2,400 → E3 runs on Azure as
specified. Coverage collapses → E3 and E4 move to a local reader, and Azure
keeps only the judge.

## What this costs

| Experiment | Calls | Approx. metered tokens | Wall clock at 1 RPM | Credentials |
| --- | ---: | ---: | ---: | --- |
| E0 acquire the artifact | 0 | 0 | — | none |
| E1 evidence partition | 0 | 0 | — | none |
| E2 provenance audit | 0 | 0 | — | none |
| E5 ceiling coverage | 0 | 0 | — | none |
| Gate 3 qualification | 1–2 | small | minutes | paid |
| E3 prompt A/B | 60 | ~37,000 | ~1 hour | paid |
| E4 holdout | 80 | ~50,000 | ~1.5 hours | paid |
| **Pilot total** | **~142** | **~87,000** | **~3 hours** | |
| *Segment B full grid, for contrast* | *~832* | *~24,000,000* | *~17 days* | *not runnable here* |

The pilot fits inside this quota. Segment B, as designed in `docs/segment-b.md`,
does not and cannot — its raw lanes exceed the per-call ceiling by 4x to 32x, so
the contrast row is there to be honest about scale, not as a plan.

Prices are deliberately absent. They are read from the deployment at approval
time, and every call carries `--max-cost` as the real-time guard.

## Order of execution

0. **Free. Done 2026-08-20.** PR #22 cherry-picked as `10c1755` and the working
   tree renormalised to LF. Suite 134 passed, `ruff check` clean, `smoke.py`
   exit 0. Gate 1 verifies the manifest at `sha256:3806648c…`; Gate 2's plan
   emits `fits_cost_cap: true` at a reserved `$0.00039345`, credential-free.
   `longmem.py run --manifest` is now available, which E0 and E5 both need.
1. **Free, but externally blocked.** E0 — request the memory artifact and the
   coverage answer. Nothing downstream starts without the artifact, and this
   endpoint cannot substitute for it.
2. **Free.** E5 and E1 as one sweep, then E2 on the artifact. All three read only
   the artifact and locally generated contexts.
3. **Decision.** If E1 falsifies the evidence claim, or E2 confirms the
   attribution gap, stop and write the artifact-contract proposal. The prompt
   experiment is the wrong fix and would paper over a structural one.
4. **Gated, and the cheapest decision in the plan.** Segment A Gate 3 — one live
   qualification call, plus one deliberately reader-sized probe. Together they
   settle four things for a fraction of a cent: the strict-schema contract,
   whether TPM enforcement refuses rather than queues, whether a reasoning model
   returns content at `--max-tokens 10`, and the deployment's context window.
   Everything below branches on its result.
5. **Free.** Stand up a local OpenAI-compatible reader and extractor, unless E5
   and Gate 3 both come back clean. This is Segment C infrastructure and the lab
   already supports it — `OPENAI_API_KEY=EMPTY --base-url
   http://127.0.0.1:8001/v1`. It removes the quota from every within-reader
   contrast at once.
6. **Paid or local.** E3, then E4 only if E3 produces a winner, driven at one call
   per minute if it runs on Azure.
7. **Free.** Freeze the outcome into `docs/segment-b.md` as the reader contract.

Steps 4 and 6 each need fresh explicit approval under the controlled-resource
rule: exact endpoint and model, case window and maximum call count, prices and
hard USD cap, files written, and stop condition.

## What the fixed quota means for Segment B

This should be settled now rather than discovered later. Segment B's ladder
cannot run on this deployment at any budget point, because rungs 0 through 4 all
require at least one whole raw session in the context and one session alone is
~4x the per-call ceiling. Dropping B4, or B3 and B4, does not help — the
constraint is per call, not per grid.

So Segment B has exactly three futures, and choosing between them is a decision
for an owner, not for this document:

1. **A different endpoint** for the reader. The ladder is unchanged.
2. **A local open-weight reader**, which makes the ladder internally valid and
   free, at the cost of comparability with published LongMemEval numbers — the
   thing the official prompt was preserved to protect.
3. **A redefined Segment B** whose lanes are all memory-shaped and fit under
   1,000 tokens. That is a different and much narrower claim than the one
   `docs/segment-b.md` currently makes, and the raw-retrieval controls it drops
   are precisely the controls that make the treatment meaningful.

The pilot is worth running under all three, because its questions are
within-reader contrasts that any of these readers can answer.

## Provenance

Every measured figure in this document was produced on 2026-08-20 against the
pinned corpus at `data/longmemeval_s_cleaned.json`
(`sha256:d6f21ea9…`, 277,383,467 bytes, 500 cases) and the working tree on
`docs/segment-b-design-main`. No model call was made, and no credential was read.

- Judge and reader prompt sizes come from calling `qa._judge_prompt()` and
  `qa._reader_prompt()` directly, tokenised with `tiktoken` `o200k_base`, over the
  30 cases in `manifests/longmemeval-s-balanced-30-v1.json`
  (`sha256:3806648c…`, read from `ef35b97`). The judge figure assumes a
  hypothesis at the full 300-token reader cap, which is its worst case.
- The characters-per-token range is measured over the real selected contexts in
  `runs/segment-a-lexical-30-contexts.jsonl`, not assumed.
- The exact-value answer stratum is counted over all 500 gold answers.
- Gold answers are short: median 11 characters, mean 52.

The 429 and empty-response hazards are read from `model.py` (`retries=0`,
`UsageLimits(request_limit=1)`, the `APIStatusError` mapping, and the empty-string
check in `complete()`) and from `qa.py`'s pending-marker lifecycle, where
`_clear_pending` runs only after a record is written. Neither has been observed
live; Gate 3 is what would observe them.

The unmeasured figures are marked with `~` in the tables above: the memory-lane
and raw-session reader sizes, and the per-session extraction size. They are
carried over from `docs/segment-b.md` and `runs/HANDOFF.md` and are order-of-
magnitude only — but each is 4x or more over the ceiling, so no plausible
measurement error changes the verdict.

## Open decisions

1. **Can the collaborator produce the memory artifact, and how many cases does it
   cover?** Everything is downstream of the artifact, and the pilot's ability to
   *confirm* a result rather than merely observe one is downstream of its
   coverage. If coverage is thin, decide immediately whether he extends the
   extraction or whether extraction moves local — that choice does not get
   cheaper by waiting for E3.
2. **Does the memory record gain a role field?** Pending E2. If yes, it is a
   change to the artifact contract and needs its own proposal, and Segment B's
   rung 5 cannot be run until it lands.
3. **Which of the three Segment B futures above?** This is now the largest open
   question in the project, and it is upstream of the reader-model decision
   rather than downstream of it.
4. **Is a local reader acceptable for the pilot?** Recommended yes. Every pilot
   question is a within-reader contrast on byte-identical contexts, so a local
   model answers them validly and without a quota. Only a headline number needs a
   comparable reader, and the pilot does not produce one.
5. **Where does extraction run?** It cannot run on Azure. If E0 delivers an
   artifact with enough coverage this is deferred; if it delivers a thin one or
   none, a local extractor is required before the memory lane exists at all.
