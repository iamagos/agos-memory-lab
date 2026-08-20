from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Protocol, TypeVar


_SCHEMA = "agos-memory-lab-case-manifest-v1"
_METHOD = "sha256-ranked-stratified-v1"
CaseT = TypeVar("CaseT", bound="CaseLike")


class CaseLike(Protocol):
  question_id: str
  question_type: str
  answer_session_ids: tuple[str, ...]


class ManifestError(Exception):
  pass


def select(
  path: Path,
  cases: tuple[CaseT, ...],
  *,
  benchmark: dict[str, Any],
  dataset: dict[str, Any],
) -> tuple[tuple[CaseT, ...], dict[str, Any]]:
  try:
    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=_invalid_constant)
  except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
    raise ManifestError("case_manifest_invalid") from exc
  root = _object(value, {"schema", "benchmark", "dataset", "selection"})
  if root["schema"] != _SCHEMA:
    raise ManifestError("case_manifest_schema_invalid")
  if root["benchmark"] != benchmark or root["dataset"] != _dataset_identity(dataset):
    raise ManifestError("case_manifest_source_mismatch")
  selection = _object(
    root["selection"],
    {
      "revision",
      "seed",
      "method",
      "cases_per_type",
      "abstention_by_type",
      "question_types",
      "question_ids",
    },
  )
  revision = _text(selection["revision"])
  seed = _text(selection["seed"])
  if selection["method"] != _METHOD:
    raise ManifestError("case_manifest_method_invalid")
  cases_per_type = _count(selection["cases_per_type"], positive=True)
  question_types = _types(selection["question_types"], cases_per_type=cases_per_type)
  abstention_by_type = _abstentions(
    selection["abstention_by_type"],
    question_types=question_types,
    cases_per_type=cases_per_type,
  )
  question_ids = _ids(selection["question_ids"])
  by_id = {case.question_id: case for case in cases}
  if len(by_id) != len(cases) or any(question_id not in by_id for question_id in question_ids):
    raise ManifestError("case_manifest_question_mismatch")
  selected = tuple(by_id[question_id] for question_id in question_ids)
  actual_types = Counter(case.question_type for case in selected)
  if actual_types != Counter(question_types):
    raise ManifestError("case_manifest_strata_mismatch")
  actual_abstentions = Counter(
    case.question_type for case in selected if case.question_id.endswith("_abs")
  )
  if any(actual_abstentions[question_type] != abstention_by_type[question_type] for question_type in question_types):
    raise ManifestError("case_manifest_abstention_mismatch")
  identity = {
    "schema": _SCHEMA,
    "file": path.name,
    "sha256": hashlib.sha256(raw).hexdigest(),
    "revision": revision,
    "seed": seed,
    "method": _METHOD,
    "cases": len(selected),
    "cases_per_type": cases_per_type,
    "abstention_by_type": abstention_by_type,
    "question_types": question_types,
  }
  return selected, identity


def build(
  cases: tuple[CaseT, ...],
  *,
  benchmark: dict[str, Any],
  dataset: dict[str, Any],
  revision: str,
  seed: str,
  cases_per_type: int,
  abstention_per_type: int,
) -> dict[str, Any]:
  revision = _text(revision)
  seed = _text(seed)
  cases_per_type = _count(cases_per_type, positive=True)
  abstention_per_type = _count(abstention_per_type, positive=False)
  if abstention_per_type > cases_per_type:
    raise ManifestError("case_manifest_strata_invalid")
  groups: dict[str, list[CaseT]] = defaultdict(list)
  for case in cases:
    groups[case.question_type].append(case)
  selected_ids: set[str] = set()
  abstention_by_type: dict[str, int] = {}
  for question_type, group in sorted(groups.items()):
    abstention = [case for case in group if case.question_id.endswith("_abs")]
    answerable = [case for case in group if not case.question_id.endswith("_abs")]
    abstention_count = min(abstention_per_type, len(abstention), cases_per_type)
    abstention_by_type[question_type] = abstention_count
    selected = _rank(abstention, seed=seed)[:abstention_count]
    selected += _rank(answerable, seed=seed)[: cases_per_type - abstention_count]
    if len(selected) != cases_per_type:
      raise ManifestError(f"case_manifest_stratum_too_small:{question_type}")
    selected_ids.update(case.question_id for case in selected)
  question_ids = [case.question_id for case in cases if case.question_id in selected_ids]
  return {
    "schema": _SCHEMA,
    "benchmark": benchmark,
    "dataset": _dataset_identity(dataset),
    "selection": {
      "revision": revision,
      "seed": seed,
      "method": _METHOD,
      "cases_per_type": cases_per_type,
      "abstention_by_type": abstention_by_type,
      "question_types": {question_type: cases_per_type for question_type in sorted(groups)},
      "question_ids": question_ids,
    },
  }


def main() -> None:
  import longmem

  parser = argparse.ArgumentParser(description="Create or verify a frozen LongMemEval case manifest.")
  commands = parser.add_subparsers(dest="command", required=True)
  create = commands.add_parser("create")
  create.add_argument("--dataset", choices=tuple(longmem._DATASETS), default="s")
  create.add_argument("--data", type=Path, default=Path("data"))
  create.add_argument("--revision", required=True)
  create.add_argument("--seed", required=True)
  create.add_argument("--cases-per-type", type=int, default=5)
  create.add_argument("--abstention-per-type", type=int, default=1)
  verify = commands.add_parser("verify")
  verify.add_argument("--dataset", choices=tuple(longmem._DATASETS), default="s")
  verify.add_argument("--data", type=Path, default=Path("data"))
  verify.add_argument("--manifest", type=Path, required=True)
  args = parser.parse_args()
  spec = longmem._DATASETS[args.dataset]
  dataset = {
    "repository": longmem._DATASET_REPOSITORY,
    "revision": longmem._DATASET_REVISION,
    "path": args.data / spec["file"],
    "sha256": spec["sha256"],
    "size": spec["size"],
  }
  benchmark = {
    "repository": longmem._BENCHMARK_REPOSITORY,
    "revision": longmem._BENCHMARK_REVISION,
  }
  try:
    longmem._verify_file(dataset["path"], sha256=dataset["sha256"], size=dataset["size"])
    cases = longmem._load(dataset["path"])
    if args.command == "create":
      value = build(
        cases,
        benchmark=benchmark,
        dataset=dataset,
        revision=args.revision,
        seed=args.seed,
        cases_per_type=args.cases_per_type,
        abstention_per_type=args.abstention_per_type,
      )
    else:
      _, value = select(args.manifest, cases, benchmark=benchmark, dataset=dataset)
    print(json.dumps(value, indent=2, sort_keys=True))
  except (ManifestError, longmem.LongMemError) as exc:
    raise SystemExit(str(exc)) from exc


def _rank(cases: list[CaseT], *, seed: str) -> list[CaseT]:
  return sorted(
    cases,
    key=lambda case: hashlib.sha256(f"{seed}:{case.question_id}".encode("utf-8")).hexdigest(),
  )


def _dataset_identity(dataset: dict[str, Any]) -> dict[str, Any]:
  return {
    "repository": dataset["repository"],
    "revision": dataset["revision"],
    "file": Path(dataset["path"]).name,
    "sha256": dataset["sha256"],
    "size": dataset["size"],
  }


def _object(value: Any, keys: set[str]) -> dict[str, Any]:
  if not isinstance(value, dict) or set(value) != keys:
    raise ManifestError("case_manifest_invalid")
  return value


def _text(value: Any) -> str:
  if not isinstance(value, str) or not value or value != value.strip():
    raise ManifestError("case_manifest_invalid")
  return value


def _count(value: Any, *, positive: bool) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < int(positive):
    raise ManifestError("case_manifest_strata_invalid")
  return value


def _types(value: Any, *, cases_per_type: int) -> dict[str, int]:
  if not isinstance(value, dict) or not value:
    raise ManifestError("case_manifest_strata_invalid")
  normalized = {_text(key): _count(count, positive=True) for key, count in value.items()}
  if any(count != cases_per_type for count in normalized.values()):
    raise ManifestError("case_manifest_strata_invalid")
  return normalized


def _abstentions(
  value: Any,
  *,
  question_types: dict[str, int],
  cases_per_type: int,
) -> dict[str, int]:
  if not isinstance(value, dict) or set(value) != set(question_types):
    raise ManifestError("case_manifest_strata_invalid")
  normalized = {_text(key): _count(count, positive=False) for key, count in value.items()}
  if any(count > cases_per_type for count in normalized.values()):
    raise ManifestError("case_manifest_strata_invalid")
  return normalized


def _ids(value: Any) -> tuple[str, ...]:
  if not isinstance(value, list):
    raise ManifestError("case_manifest_question_invalid")
  normalized = tuple(_text(question_id) for question_id in value)
  if not normalized or len(normalized) != len(set(normalized)):
    raise ManifestError("case_manifest_question_invalid")
  return normalized


def _invalid_constant(_: str) -> None:
  raise ValueError


if __name__ == "__main__":
  main()
