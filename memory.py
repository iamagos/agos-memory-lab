from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from agos_memory.admit import admit
from agos_memory.support import source_digest
from agos_memory.types import (
  Accept,
  AdmissionRules,
  ExistingRecord,
  Partition,
  PartitionRule,
  Proposal,
  Reject,
  Replace,
  SourceDependency,
)

import longmem


_SCHEMA = "agos-memory-lab-memory-v1"
_EXTRACTOR_SCHEMA = "agos-memory-lab-extractor-v1"
_AVAILABILITY = frozenset(("released", "causal"))
_KINDS = frozenset(("event", "fact", "preference"))
_RULES = AdmissionRules(
  partition_rules=(
    PartitionRule(
      label="case",
      missing_key_reason="missing_case",
      allowed_kinds=_KINDS,
      disallowed_kind_reason="kind_not_allowed",
    ),
  ),
)


@dataclass(frozen=True, slots=True)
class Extractor:
  name: str
  revision: str
  config_json: str
  sha256: str
  size: int
  rows: tuple["ExtractorRow", ...]

  @property
  def config(self) -> dict[str, Any]:
    return json.loads(self.config_json)

  @property
  def identity(self) -> dict[str, Any]:
    return {
      "schema": _EXTRACTOR_SCHEMA,
      "name": self.name,
      "revision": self.revision,
      "config": self.config,
    }


@dataclass(frozen=True, slots=True)
class ExtractorRow:
  case_id: str
  source_id: str
  session_date: str
  source_digest: str
  ordinal: int
  proposal_id: str
  kind: str
  text: str
  confidence: float
  supersedes: tuple[str, ...]
  expires_days: int | None


@dataclass(frozen=True, slots=True)
class Source:
  source_id: str
  date: str
  at: datetime
  content: str


@dataclass(frozen=True, slots=True)
class SourceCase:
  case_id: str
  cutoff: datetime
  sessions: tuple[Source, ...]


@dataclass(frozen=True, slots=True)
class Record:
  record_id: str
  case_id: str
  kind: str
  text: str
  confidence: float
  expires_at: datetime | None
  source_ref: str
  source_date: str
  source: SourceDependency


@dataclass(frozen=True, slots=True)
class Admission:
  cases: int
  proposals: int
  outcomes: tuple[tuple[str, int], ...]
  reasons: tuple[tuple[str, int], ...]

  def value(self) -> dict[str, Any]:
    return {
      "cases": self.cases,
      "proposals": self.proposals,
      "outcomes": dict(self.outcomes),
      "reasons": dict(self.reasons),
    }


@dataclass(frozen=True, slots=True)
class Artifact:
  artifact_id: str
  run_id: str
  sha256: str
  kernel: str
  benchmark_repository: str
  benchmark_revision: str
  dataset_repository: str
  dataset_revision: str
  dataset_sha256: str
  dataset_size: int
  extractor_name: str
  extractor_revision: str
  extractor_config_json: str
  extractor_sha256: str
  extractor_size: int
  availability: str
  admission: Admission
  records: tuple[Record, ...]

  @property
  def identity(self) -> dict[str, Any]:
    return {
      "artifact_id": self.artifact_id,
      "compiler_run_id": self.run_id,
      "sha256": self.sha256,
      "kernel": self.kernel,
      "extractor": {
        "name": self.extractor_name,
        "revision": self.extractor_revision,
        "config": json.loads(self.extractor_config_json),
        "sha256": self.extractor_sha256,
        "size": self.extractor_size,
      },
      "admission": self.admission.value(),
    }


class MemoryCompileError(Exception):
  pass


def main() -> None:
  parser = _parser()
  args = parser.parse_args()
  try:
    source = longmem._source(args)
    longmem._verify_file(source["path"], sha256=source["sha256"], size=source["size"])
    extractor = _load_extractor(args.extractor, sha256=args.extractor_sha256)
    artifact = _compile(
      _sources(longmem._load(source["path"])),
      dataset_repository=source["repository"],
      dataset_revision=source["revision"],
      benchmark_repository=longmem._BENCHMARK_REPOSITORY,
      benchmark_revision=longmem._BENCHMARK_REVISION,
      extractor=extractor,
    )
    receipt = _receipt(source=source, extractor=extractor, artifact=artifact)
    out = args.out or Path("runs") / f"memory-{receipt['artifact']['artifact_id'][:12]}.json"
    longmem._write(out, receipt)
    print(
      json.dumps(
        {
          "artifact_id": receipt["artifact"]["artifact_id"],
          "out": str(out),
          "run_id": receipt["run_id"],
          "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
          "summary": receipt["artifact"]["summary"],
        },
        indent=2,
        sort_keys=True,
      )
    )
  except (MemoryCompileError, longmem.LongMemError, ValueError) as exc:
    raise SystemExit(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Compile source-linked LongMemEval memories.")
  source = parser.add_mutually_exclusive_group()
  source.add_argument("--dataset", choices=tuple(longmem._DATASETS), help="Pinned corpus; defaults to s.")
  source.add_argument("--file", type=Path, help="Custom LongMemEval-shaped JSON file.")
  parser.add_argument("--sha256", help="Required SHA-256 for --file.")
  parser.add_argument("--revision", help="Required immutable identity for --file.")
  parser.add_argument("--data", type=Path, default=Path("data"))
  parser.add_argument("--extractor", type=Path, required=True, help="Frozen extractor JSONL.")
  parser.add_argument("--extractor-sha256", required=True, help="Expected extractor SHA-256.")
  parser.add_argument("--out", type=Path)
  return parser


def load(path: Path, *, sha256: str) -> Artifact:
  expected = _sha256(sha256, error="memory_artifact_sha256_invalid")
  try:
    raw = path.read_bytes()
  except OSError as exc:
    raise MemoryCompileError(f"memory_artifact_read_failed:{exc}") from exc
  actual = hashlib.sha256(raw).hexdigest()
  if actual != expected:
    raise MemoryCompileError("memory_artifact_sha256_mismatch")
  try:
    value = json.loads(raw, parse_constant=_invalid_constant)
  except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
    raise MemoryCompileError("memory_artifact_json_invalid") from exc
  receipt = _object(
    value,
    keys={"schema", "inputs", "artifact", "run_id", "receipt_sha256"},
    error="memory_artifact_receipt_invalid",
  )
  if receipt["schema"] != "agos-memory-lab-memory-compile-v1":
    raise MemoryCompileError("memory_artifact_receipt_invalid")
  receipt_sha256 = _sha256(
    receipt["receipt_sha256"],
    error="memory_artifact_receipt_digest_invalid",
  )
  receipt_body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
  if receipt_sha256 != _digest(receipt_body):
    raise MemoryCompileError("memory_artifact_receipt_digest_mismatch")
  run_id = _sha256(receipt["run_id"], error="memory_artifact_run_identity_invalid")
  semantic = {key: item for key, item in receipt.items() if key not in {"receipt_sha256", "run_id"}}
  if run_id != _digest(semantic):
    raise MemoryCompileError("memory_artifact_run_identity_mismatch")
  try:
    return _artifact(receipt, sha256=actual)
  except MemoryCompileError:
    raise
  except ValueError as exc:
    raise MemoryCompileError("memory_artifact_value_invalid") from exc


def _artifact(receipt: dict[str, Any], *, sha256: str) -> Artifact:
  inputs = _object(
    receipt["inputs"],
    keys={"dataset", "extractor"},
    error="memory_artifact_inputs_invalid",
  )
  dataset_input = _object(
    inputs["dataset"],
    keys={"repository", "revision", "sha256", "size"},
    error="memory_artifact_dataset_invalid",
  )
  extractor_input = _object(
    inputs["extractor"],
    keys={"sha256", "size"},
    error="memory_artifact_extractor_invalid",
  )
  _sha256(dataset_input["sha256"], error="memory_artifact_dataset_invalid")
  _size(dataset_input["size"], error="memory_artifact_dataset_invalid")
  _sha256(extractor_input["sha256"], error="memory_artifact_extractor_invalid")
  _size(extractor_input["size"], error="memory_artifact_extractor_invalid")

  artifact = _object(
    receipt["artifact"],
    keys={
      "schema",
      "kernel",
      "benchmark",
      "dataset",
      "extractor",
      "rules",
      "summary",
      "cases",
      "artifact_id",
    },
    error="memory_artifact_invalid",
  )
  if artifact["schema"] != _SCHEMA:
    raise MemoryCompileError("memory_artifact_invalid")
  artifact_id = _sha256(artifact["artifact_id"], error="memory_artifact_identity_invalid")
  artifact_body = {key: item for key, item in artifact.items() if key != "artifact_id"}
  if artifact_id != _digest(artifact_body):
    raise MemoryCompileError("memory_artifact_identity_mismatch")
  kernel = _text(artifact["kernel"], error="memory_artifact_kernel_invalid")
  benchmark = _object(
    artifact["benchmark"],
    keys={"repository", "revision"},
    error="memory_artifact_benchmark_invalid",
  )
  benchmark_repository = _text(
    benchmark["repository"],
    error="memory_artifact_benchmark_invalid",
  )
  benchmark_revision = _text(
    benchmark["revision"],
    error="memory_artifact_benchmark_invalid",
  )
  dataset = _object(
    artifact["dataset"],
    keys={"repository", "revision"},
    error="memory_artifact_dataset_invalid",
  )
  repository = _text(dataset["repository"], error="memory_artifact_dataset_invalid")
  revision = _text(dataset["revision"], error="memory_artifact_dataset_invalid")
  if dataset_input["repository"] != repository or dataset_input["revision"] != revision:
    raise MemoryCompileError("memory_artifact_dataset_mismatch")
  extractor = _extractor_identity(artifact["extractor"])
  records, admission = _records(
    artifact["cases"],
    extractor=extractor,
    dataset_repository=repository,
    dataset_revision=revision,
  )
  summary = _object(
    artifact["summary"],
    keys={"cases", "proposals", "outcomes", "active_records"},
    error="memory_artifact_summary_invalid",
  )
  expected_summary = {
    "cases": admission.cases,
    "proposals": admission.proposals,
    "outcomes": dict(admission.outcomes),
    "active_records": len(records),
  }
  if summary != expected_summary:
    raise MemoryCompileError("memory_artifact_summary_mismatch")
  return Artifact(
    artifact_id=artifact_id,
    run_id=receipt["run_id"],
    sha256=sha256,
    kernel=kernel,
    benchmark_repository=benchmark_repository,
    benchmark_revision=benchmark_revision,
    dataset_repository=repository,
    dataset_revision=revision,
    dataset_sha256=dataset_input["sha256"],
    dataset_size=dataset_input["size"],
    extractor_name=extractor["name"],
    extractor_revision=extractor["revision"],
    extractor_config_json=_canonical(extractor["config"]).decode(),
    extractor_sha256=extractor_input["sha256"],
    extractor_size=extractor_input["size"],
    availability=_availability(extractor["config"]),
    admission=admission,
    records=records,
  )


def _extractor_identity(value: Any) -> dict[str, Any]:
  extractor = _object(
    value,
    keys={"schema", "name", "revision", "config"},
    error="memory_artifact_extractor_invalid",
  )
  if extractor["schema"] != _EXTRACTOR_SCHEMA or not isinstance(extractor["config"], dict):
    raise MemoryCompileError("memory_artifact_extractor_invalid")
  _text(extractor["name"], error="memory_artifact_extractor_invalid")
  _text(extractor["revision"], error="memory_artifact_extractor_invalid")
  _canonical(extractor["config"])
  return extractor


def _records(
  values: Any,
  *,
  extractor: dict[str, Any],
  dataset_repository: str,
  dataset_revision: str,
) -> tuple[tuple[Record, ...], Admission]:
  if not isinstance(values, list):
    raise MemoryCompileError("memory_artifact_cases_invalid")
  case_ids: list[str] = []
  records: list[Record] = []
  seen: set[str] = set()
  outcomes = {"accept": 0, "reject": 0, "replace": 0}
  reasons: dict[str, int] = {}
  proposals = 0
  for value in values:
    case = _object(
      value,
      keys={"case_id", "cutoff", "decisions", "active_record_ids"},
      error="memory_artifact_case_invalid",
    )
    case_id = _text(case["case_id"], error="memory_artifact_case_invalid")
    case_ids.append(case_id)
    if _instant(case["cutoff"]) is None:
      raise MemoryCompileError("memory_artifact_case_cutoff_invalid")
    decisions = case["decisions"]
    if not isinstance(decisions, list):
      raise MemoryCompileError("memory_artifact_decisions_invalid")
    accepted: dict[str, dict[str, Any]] = {}
    for decision in decisions:
      if not isinstance(decision, dict) or decision.get("outcome") not in {"accept", "replace", "reject"}:
        raise MemoryCompileError("memory_artifact_decision_invalid")
      outcome = decision["outcome"]
      outcomes[outcome] += 1
      proposals += 1
      if outcome == "reject":
        reason = _text(decision.get("reason"), error="memory_artifact_decision_invalid")
        reasons[reason] = reasons.get(reason, 0) + 1
      record_id = decision.get("record_id")
      if outcome != "reject":
        record_id = _text(record_id, error="memory_artifact_record_identity_invalid")
        if record_id in accepted:
          raise MemoryCompileError("memory_artifact_record_identity_duplicated")
        accepted[record_id] = decision
    active = case["active_record_ids"]
    if not isinstance(active, list) or any(not isinstance(record_id, str) for record_id in active):
      raise MemoryCompileError("memory_artifact_active_records_invalid")
    if active != sorted(set(active)):
      raise MemoryCompileError("memory_artifact_active_records_invalid")
    for record_id in active:
      decision = accepted.get(record_id)
      if decision is None or record_id in seen:
        raise MemoryCompileError("memory_artifact_active_record_missing")
      seen.add(record_id)
      records.append(
        _record(
          case_id=case_id,
          decision=decision,
          extractor=extractor,
          dataset_repository=dataset_repository,
          dataset_revision=dataset_revision,
        )
      )
  if case_ids != sorted(set(case_ids)):
    raise MemoryCompileError("memory_artifact_case_identity_invalid")
  return tuple(records), Admission(
    cases=len(case_ids),
    proposals=proposals,
    outcomes=tuple(outcomes.items()),
    reasons=tuple(sorted(reasons.items())),
  )


def _record(
  *,
  case_id: str,
  decision: dict[str, Any],
  extractor: dict[str, Any],
  dataset_repository: str,
  dataset_revision: str,
) -> Record:
  outcome = decision["outcome"]
  keys = {
    "proposal_id",
    "ordinal",
    "source",
    "proposal",
    "outcome",
    "expires_at",
    "record_id",
  }
  if outcome == "replace":
    keys.add("replaced_record_ids")
  if set(decision) != keys:
    raise MemoryCompileError("memory_artifact_record_invalid")
  ordinal = decision["ordinal"]
  if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
    raise MemoryCompileError("memory_artifact_record_invalid")
  _text(decision["proposal_id"], error="memory_artifact_record_invalid")
  record_id = _text(decision["record_id"], error="memory_artifact_record_identity_invalid")
  record = {key: item for key, item in decision.items() if key not in {"proposal_id", "record_id"}}
  expected_id = "memory:" + _digest(
    {
      "schema": "agos-memory-lab-record-v1",
      "extractor": extractor,
      "decision": record,
    }
  )
  if record_id != expected_id:
    raise MemoryCompileError("memory_artifact_record_identity_mismatch")
  source_ref, source_date, source = _record_source(
    decision["source"],
    case_id=case_id,
    dataset_repository=dataset_repository,
    dataset_revision=dataset_revision,
  )
  proposal = _record_proposal(decision["proposal"], case_id=case_id, source_ref=source_ref)
  replaced = decision.get("replaced_record_ids", [])
  if not isinstance(replaced, list) or replaced != sorted(set(replaced)):
    raise MemoryCompileError("memory_artifact_replacement_invalid")
  if (outcome == "accept" and proposal.supersedes) or (
    outcome == "replace" and tuple(replaced) != proposal.supersedes
  ):
    raise MemoryCompileError("memory_artifact_replacement_mismatch")
  return Record(
    record_id=record_id,
    case_id=case_id,
    kind=proposal.kind,
    text=proposal.text,
    confidence=proposal.confidence,
    expires_at=_instant(decision["expires_at"]),
    source_ref=source_ref,
    source_date=source_date,
    source=source,
  )


def _record_source(
  value: Any,
  *,
  case_id: str,
  dataset_repository: str,
  dataset_revision: str,
) -> tuple[str, str, SourceDependency]:
  source = _object(
    value,
    keys={"ref", "owner", "revision", "fragment", "kind", "date", "digest"},
    error="memory_artifact_source_invalid",
  )
  source_ref = _text(source["ref"], error="memory_artifact_source_invalid")
  body = {key: item for key, item in source.items() if key != "ref"}
  if source_ref != f"source:{_digest(body)}":
    raise MemoryCompileError("memory_artifact_source_identity_mismatch")
  dependency = SourceDependency(
    owner=source["owner"],
    revision=source["revision"],
    fragment=source["fragment"],
    kind=source["kind"],
    digest=source["digest"],
  )
  if (
    dependency.owner != f"{dataset_repository}#{case_id}"
    or dependency.revision != dataset_revision
    or dependency.kind != "session"
  ):
    raise MemoryCompileError("memory_artifact_source_scope_mismatch")
  return source_ref, _text(source["date"], error="memory_artifact_source_invalid"), dependency


def _record_proposal(value: Any, *, case_id: str, source_ref: str) -> Proposal:
  proposal = _object(
    value,
    keys={
      "partition",
      "kind",
      "text",
      "confidence",
      "source_refs",
      "supersedes",
      "expires_days",
    },
    error="memory_artifact_proposal_invalid",
  )
  partition = _object(
    proposal["partition"],
    keys={"label", "key"},
    error="memory_artifact_proposal_invalid",
  )
  if partition != {"label": "case", "key": case_id}:
    raise MemoryCompileError("memory_artifact_proposal_scope_mismatch")
  source_refs = proposal["source_refs"]
  supersedes = proposal["supersedes"]
  if source_refs != [source_ref] or not isinstance(supersedes, list):
    raise MemoryCompileError("memory_artifact_proposal_source_mismatch")
  result = Proposal(
    partition=Partition("case", case_id),
    kind=proposal["kind"],
    text=proposal["text"],
    confidence=proposal["confidence"],
    source_refs=tuple(source_refs),
    supersedes=tuple(supersedes),
    expires_days=proposal["expires_days"],
  )
  if result.text != " ".join(result.text.split()):
    raise MemoryCompileError("memory_artifact_proposal_not_normalized")
  return result


def _instant(value: Any) -> datetime | None:
  if value is None:
    return None
  if not isinstance(value, str):
    raise MemoryCompileError("memory_artifact_expiry_invalid")
  try:
    result = datetime.fromisoformat(value)
  except ValueError as exc:
    raise MemoryCompileError("memory_artifact_expiry_invalid") from exc
  if result.tzinfo is None or result.utcoffset() is None:
    raise MemoryCompileError("memory_artifact_expiry_invalid")
  return result


def _sources(cases: tuple[Any, ...]) -> tuple[SourceCase, ...]:
  return tuple(
    SourceCase(
      case_id=case.question_id,
      cutoff=case.asked_at,
      sessions=tuple(
        Source(
          source_id=session.source_id,
          date=session.date,
          at=session.at,
          content=session.content,
        )
        for session in case.sessions
      ),
    )
    for case in cases
  )


def _compile(
  cases: tuple[SourceCase, ...],
  *,
  dataset_repository: str,
  dataset_revision: str,
  benchmark_repository: str,
  benchmark_revision: str,
  extractor: Extractor,
) -> dict[str, Any]:
  availability = _availability(extractor.config)
  by_case = {case.case_id: case for case in cases}
  sessions_by_case = {
    case_id: {session.source_id: session for session in case.sessions}
    for case_id, case in by_case.items()
  }
  rows_by_case: dict[str, list[ExtractorRow]] = {}
  for row in extractor.rows:
    rows_by_case.setdefault(row.case_id, []).append(row)
  _validate_rows(
    extractor.rows,
    by_case=by_case,
    sessions_by_case=sessions_by_case,
    availability=availability,
  )
  aliases = _aliases(extractor.rows)
  extractor_identity = extractor.identity
  compiled = []
  counts = {"accept": 0, "reject": 0, "replace": 0}

  for case_id in sorted(rows_by_case):
    case = by_case[case_id]
    source_order = {session.source_id: index for index, session in enumerate(case.sessions)}
    sessions = sessions_by_case[case_id]
    rows = sorted(
      rows_by_case[case_id],
      key=lambda row: (sessions[row.source_id].at, source_order[row.source_id], row.ordinal),
    )
    positions = {row.proposal_id: index for index, row in enumerate(rows)}
    accepted: dict[str, str] = {}
    active: dict[str, ExistingRecord] = {}
    decisions = []

    for index, row in enumerate(rows):
      session = sessions[row.source_id]
      source = _source(
        case_id=case_id,
        dataset_repository=dataset_repository,
        dataset_revision=dataset_revision,
        row=row,
        session=session,
      )
      supersedes = _supersedes(
        row,
        index=index,
        positions=positions,
        aliases=aliases,
        accepted=accepted,
      )
      proposal = Proposal(
        partition=Partition("case", case_id),
        kind=row.kind,
        text=row.text,
        confidence=row.confidence,
        source_refs=(source["ref"],),
        supersedes=supersedes,
        expires_days=row.expires_days,
      )
      outcome = admit(
        (proposal,),
        tuple(active.values()),
        rules=_RULES,
        now=session.at,
      )[0]
      decision = _decision(
        row=row,
        source=source,
        outcome=outcome,
        extractor=extractor_identity,
      )
      counts[decision["outcome"]] += 1

      if isinstance(outcome, Accept | Replace):
        record_id = decision["record_id"]
        accepted[row.proposal_id] = record_id
        if isinstance(outcome, Replace):
          for replaced_id in outcome.replaced_record_ids:
            active.pop(replaced_id)
        active[record_id] = ExistingRecord(
          record_id=record_id,
          partition=outcome.proposal.partition,
          kind=outcome.proposal.kind,
          text=outcome.proposal.text,
        )
      decisions.append(decision)

    compiled.append(
      {
        "case_id": case_id,
        "cutoff": case.cutoff.isoformat(),
        "decisions": decisions,
        "active_record_ids": sorted(active),
      }
    )

  return {
    "schema": _SCHEMA,
    "kernel": version("agos-memory"),
    "benchmark": {
      "repository": benchmark_repository,
      "revision": benchmark_revision,
    },
    "dataset": {
      "repository": dataset_repository,
      "revision": dataset_revision,
    },
    "extractor": extractor_identity,
    "rules": _rules(),
    "summary": {
      "cases": len(compiled),
      "proposals": sum(counts.values()),
      "outcomes": counts,
      "active_records": sum(len(case["active_record_ids"]) for case in compiled),
    },
    "cases": compiled,
  }


def _validate_rows(
  rows: tuple[ExtractorRow, ...],
  *,
  by_case: dict[str, SourceCase],
  sessions_by_case: dict[str, dict[str, Source]],
  availability: str,
) -> None:
  if not rows:
    raise MemoryCompileError("extractor_proposals_empty")
  all_sources = {
    source_id
    for sessions in sessions_by_case.values()
    for source_id in sessions
  }

  proposal_keys: set[tuple[str, str]] = set()
  ordinal_keys: set[tuple[str, str, int]] = set()
  for row in rows:
    case = by_case.get(row.case_id)
    if case is None:
      raise MemoryCompileError("extractor_case_missing")
    session = sessions_by_case[row.case_id].get(row.source_id)
    if session is None:
      reason = "extractor_source_scope_mismatch" if row.source_id in all_sources else "extractor_source_missing"
      raise MemoryCompileError(reason)
    if row.session_date != session.date:
      raise MemoryCompileError("extractor_source_date_mismatch")
    if row.source_digest != source_digest(session.content):
      raise MemoryCompileError("extractor_source_digest_mismatch")
    if availability == "causal" and session.at > case.cutoff:
      raise MemoryCompileError("extractor_source_future")

    proposal_key = (row.case_id, row.proposal_id)
    if proposal_key in proposal_keys:
      raise MemoryCompileError("extractor_proposal_identity_duplicated")
    proposal_keys.add(proposal_key)
    ordinal_key = (row.case_id, row.source_id, row.ordinal)
    if ordinal_key in ordinal_keys:
      raise MemoryCompileError("extractor_proposal_ordinal_duplicated")
    ordinal_keys.add(ordinal_key)


def _aliases(rows: tuple[ExtractorRow, ...]) -> dict[str, set[str]]:
  aliases: dict[str, set[str]] = {}
  for row in rows:
    aliases.setdefault(row.proposal_id, set()).add(row.case_id)
  return aliases


def _supersedes(
  row: ExtractorRow,
  *,
  index: int,
  positions: dict[str, int],
  aliases: dict[str, set[str]],
  accepted: dict[str, str],
) -> tuple[str, ...]:
  record_ids = []
  for proposal_id in row.supersedes:
    if proposal_id in accepted:
      record_ids.append(accepted[proposal_id])
    elif proposal_id in positions:
      reason = (
        "extractor_supersedes_future"
        if positions[proposal_id] >= index
        else "extractor_supersedes_unavailable"
      )
      raise MemoryCompileError(reason)
    elif proposal_id in aliases:
      raise MemoryCompileError("extractor_supersedes_scope_mismatch")
    else:
      raise MemoryCompileError("extractor_supersedes_missing")
  return tuple(sorted(record_ids))


def _source(
  *,
  case_id: str,
  dataset_repository: str,
  dataset_revision: str,
  row: ExtractorRow,
  session: Source,
) -> dict[str, str]:
  dependency = SourceDependency(
    owner=f"{dataset_repository}#{case_id}",
    revision=dataset_revision,
    fragment=session.source_id,
    kind="session",
    digest=row.source_digest,
  )
  value = {
    "owner": dependency.owner,
    "revision": dependency.revision,
    "fragment": dependency.fragment,
    "kind": dependency.kind,
    "date": row.session_date,
    "digest": dependency.digest,
  }
  return {"ref": f"source:{_digest(value)}", **value}


def _decision(
  *,
  row: ExtractorRow,
  source: dict[str, str],
  outcome: Accept | Reject | Replace,
  extractor: dict[str, Any],
) -> dict[str, Any]:
  proposal = _proposal(outcome.proposal)
  value: dict[str, Any] = {
    "proposal_id": row.proposal_id,
    "ordinal": row.ordinal,
    "source": source,
    "proposal": proposal,
  }
  if isinstance(outcome, Reject):
    return {**value, "outcome": "reject", "reason": outcome.reason}

  accepted: dict[str, Any] = {
    **value,
    "outcome": "replace" if isinstance(outcome, Replace) else "accept",
    "expires_at": outcome.expires_at.isoformat() if outcome.expires_at is not None else None,
  }
  if isinstance(outcome, Replace):
    accepted["replaced_record_ids"] = list(outcome.replaced_record_ids)
  record = {key: value for key, value in accepted.items() if key != "proposal_id"}
  record_id = "memory:" + _digest(
    {
      "schema": "agos-memory-lab-record-v1",
      "extractor": extractor,
      "decision": record,
    }
  )
  return {**accepted, "record_id": record_id}


def _proposal(proposal: Proposal) -> dict[str, Any]:
  return {
    "partition": {"label": proposal.partition.label, "key": proposal.partition.key},
    "kind": proposal.kind,
    "text": proposal.text,
    "confidence": proposal.confidence,
    "source_refs": list(proposal.source_refs),
    "supersedes": list(proposal.supersedes),
    "expires_days": proposal.expires_days,
  }


def _rules() -> dict[str, Any]:
  return {
    "min_confidence": _RULES.min_confidence,
    "max_text_chars": _RULES.max_text_chars,
    "partitions": [
      {
        "label": rule.label,
        "missing_key_reason": rule.missing_key_reason,
        "allowed_kinds": sorted(rule.allowed_kinds) if rule.allowed_kinds is not None else None,
        "disallowed_kind_reason": rule.disallowed_kind_reason,
      }
      for rule in _RULES.partition_rules
    ],
  }


def _receipt(*, source: dict[str, Any], extractor: Extractor, artifact: dict[str, Any]) -> dict[str, Any]:
  artifact_id = _digest(artifact)
  artifact = {**artifact, "artifact_id": artifact_id}
  semantic = {
    "schema": "agos-memory-lab-memory-compile-v1",
    "inputs": {
      "dataset": {
        "repository": source["repository"],
        "revision": source["revision"],
        "sha256": source["sha256"],
        "size": source["size"],
      },
      "extractor": {
        "sha256": extractor.sha256,
        "size": extractor.size,
      },
    },
    "artifact": artifact,
  }
  receipt = {**semantic, "run_id": _digest(semantic)}
  return {**receipt, "receipt_sha256": _digest(receipt)}


def _load_extractor(path: Path, *, sha256: str) -> Extractor:
  expected = _sha256(sha256, error="extractor_sha256_invalid")
  try:
    raw = path.read_bytes()
  except OSError as exc:
    raise MemoryCompileError(f"extractor_read_failed:{exc}") from exc
  actual = hashlib.sha256(raw).hexdigest()
  if actual != expected:
    raise MemoryCompileError("extractor_sha256_mismatch")
  try:
    lines = raw.decode("utf-8").splitlines()
  except UnicodeDecodeError as exc:
    raise MemoryCompileError("extractor_utf8_invalid") from exc
  if not lines or any(not line.strip() for line in lines):
    raise MemoryCompileError("extractor_line_empty")
  values = tuple(_json(line) for line in lines)
  header = _header(values[0])
  rows = tuple(_row(value) for value in values[1:])
  return Extractor(
    name=header["name"],
    revision=header["revision"],
    config_json=_canonical(header["config"]).decode(),
    sha256=actual,
    size=len(raw),
    rows=rows,
  )


def _header(value: Any) -> dict[str, Any]:
  keys = {"type", "schema", "name", "revision", "config"}
  if not isinstance(value, dict) or set(value) != keys:
    raise MemoryCompileError("extractor_header_invalid")
  if value["type"] != "extractor" or value["schema"] != _EXTRACTOR_SCHEMA:
    raise MemoryCompileError("extractor_header_invalid")
  name = _text(value["name"], error="extractor_name_invalid")
  revision = _text(value["revision"], error="extractor_revision_invalid")
  if not isinstance(value["config"], dict):
    raise MemoryCompileError("extractor_config_invalid")
  _canonical(value["config"])
  return {"name": name, "revision": revision, "config": value["config"]}


def _availability(config: dict[str, Any]) -> str:
  value = config.get("availability", "causal")
  if value not in _AVAILABILITY:
    raise MemoryCompileError("extractor_availability_invalid")
  return value


def _row(value: Any) -> ExtractorRow:
  keys = {
    "type",
    "case_id",
    "source_id",
    "session_date",
    "source_digest",
    "ordinal",
    "proposal_id",
    "kind",
    "text",
    "confidence",
    "supersedes",
    "expires_days",
  }
  if not isinstance(value, dict) or set(value) != keys or value["type"] != "proposal":
    raise MemoryCompileError("extractor_row_invalid")
  ordinal = value["ordinal"]
  if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
    raise MemoryCompileError("extractor_ordinal_invalid")
  supersedes = value["supersedes"]
  if not isinstance(supersedes, list):
    raise MemoryCompileError("extractor_supersedes_invalid")
  normalized_supersedes = tuple(
    sorted(_text(item, error="extractor_supersedes_invalid") for item in supersedes)
  )
  if len(normalized_supersedes) != len(set(normalized_supersedes)):
    raise MemoryCompileError("extractor_supersedes_duplicated")
  expires_days = value["expires_days"]
  if expires_days is not None and (
    not isinstance(expires_days, int) or isinstance(expires_days, bool)
  ):
    raise MemoryCompileError("extractor_expiry_invalid")
  confidence = value["confidence"]
  if not isinstance(confidence, int | float) or isinstance(confidence, bool):
    raise MemoryCompileError("extractor_confidence_invalid")
  return ExtractorRow(
    case_id=_text(value["case_id"], error="extractor_case_id_invalid"),
    source_id=_text(value["source_id"], error="extractor_source_id_invalid"),
    session_date=_text(value["session_date"], error="extractor_session_date_invalid"),
    source_digest=_sha256(value["source_digest"], error="extractor_source_digest_invalid", prefix=True),
    ordinal=ordinal,
    proposal_id=_text(value["proposal_id"], error="extractor_proposal_id_invalid"),
    kind=_text(value["kind"], error="extractor_kind_invalid"),
    text=_string(value["text"], error="extractor_text_invalid"),
    confidence=float(confidence),
    supersedes=normalized_supersedes,
    expires_days=expires_days,
  )


def _json(line: str) -> Any:
  try:
    return json.loads(line, parse_constant=_invalid_constant)
  except (json.JSONDecodeError, ValueError) as exc:
    raise MemoryCompileError("extractor_json_invalid") from exc


def _invalid_constant(_: str) -> None:
  raise ValueError


def _text(value: Any, *, error: str) -> str:
  if not isinstance(value, str) or not value or value != value.strip():
    raise MemoryCompileError(error)
  return value


def _string(value: Any, *, error: str) -> str:
  if not isinstance(value, str):
    raise MemoryCompileError(error)
  return value


def _object(value: Any, *, keys: set[str], error: str) -> dict[str, Any]:
  if not isinstance(value, dict) or set(value) != keys:
    raise MemoryCompileError(error)
  return value


def _size(value: Any, *, error: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 1:
    raise MemoryCompileError(error)
  return value


def _sha256(value: Any, *, error: str, prefix: bool = False) -> str:
  expected_length = 71 if prefix else 64
  start = 7 if prefix else 0
  if (
    not isinstance(value, str)
    or len(value) != expected_length
    or (prefix and not value.startswith("sha256:"))
    or any(character not in "0123456789abcdef" for character in value[start:])
  ):
    raise MemoryCompileError(error)
  return value


def _digest(value: Any) -> str:
  return hashlib.sha256(_canonical(value)).hexdigest()


def _canonical(value: Any) -> bytes:
  try:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
  except (TypeError, ValueError) as exc:
    raise MemoryCompileError("canonical_value_invalid") from exc


if __name__ == "__main__":
  main()
