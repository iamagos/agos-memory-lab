from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from statistics import fmean
from typing import Any

import longmem
import memory
import qa


_FAILURES = (
  "fact-not-extracted",
  "fact-not-retrieved",
  "lost-in-packing",
  "reader-hedge",
  "reader-unsupported",
  "failed-abstention",
  "over-abstention",
  "judge-disagreement",
)


class ProbeError(Exception):
  pass


def main() -> None:
  parser = _parser()
  args = parser.parse_args()
  try:
    if args.command == "check-operand":
      value = _check_operand(args)
    elif args.command == "answerability":
      value = _answerability(args)
    elif args.command == "code-failures":
      value = _code_failures(args)
    else:
      parser.error("command_required")
      return
    _write(args.out, value)
    print(json.dumps({"out": str(args.out), "run_id": value["run_id"], "summary": value["summary"]}, sort_keys=True))
  except (ProbeError, qa.QAError, longmem.LongMemError, memory.MemoryCompileError) as exc:
    raise SystemExit(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run deterministic LongMemEval gap probes.")
  commands = parser.add_subparsers(dest="command")

  operand = commands.add_parser("check-operand", help="Check literal gold operands in selected contexts.")
  operand.add_argument("--contexts", type=Path, required=True)
  operand.add_argument("--out", type=Path, required=True)
  _reference_arguments(operand)

  answerability = commands.add_parser("answerability", help="Measure retrieval-score answerability signals.")
  answerability.add_argument("--receipt", type=Path, required=True)
  answerability.add_argument("--out", type=Path, required=True)
  _reference_arguments(answerability)

  failures = commands.add_parser("code-failures", help="Mechanically code judged-wrong cases.")
  failures.add_argument("--receipt", type=Path, required=True)
  failures.add_argument("--contexts", type=Path, required=True)
  failures.add_argument("--hypotheses", type=Path, required=True)
  failures.add_argument("--evaluations", type=Path, required=True)
  failures.add_argument("--comparison-evaluations", type=Path)
  failures.add_argument("--artifact", type=Path)
  failures.add_argument("--artifact-sha256")
  failures.add_argument("--out", type=Path, required=True)
  _reference_arguments(failures)
  return parser


def _reference_arguments(parser: argparse.ArgumentParser) -> None:
  source = parser.add_mutually_exclusive_group()
  source.add_argument("--dataset", choices=tuple(longmem._DATASETS), default="s")
  source.add_argument("--references", type=Path)
  parser.add_argument("--sha256")
  parser.add_argument("--revision")
  parser.add_argument("--data", type=Path, default=Path("data"))


def _check_operand(args: argparse.Namespace) -> dict[str, Any]:
  contexts, contexts_sha256 = qa._contexts(args.contexts)
  references, source = qa._references(args)
  by_id = {reference.question_id: reference for reference in references}
  corpus = {case.question_id: case for case in _corpus(args)}
  if any(context.question_id not in by_id or context.question_id not in corpus for context in contexts):
    raise ProbeError("operand_context_reference_mismatch")
  cases = []
  for context in contexts:
    reference = by_id[context.question_id]
    present = _contains(context.context, reference.answer)
    full_history_present = None if reference.abstention else _contains_any(
      tuple(
        f"Session Date: {session.date}\nSession Content:\n{session.content}"
        for session in corpus[context.question_id].sessions
      ),
      reference.answer,
    )
    status = "not-applicable" if reference.abstention else (
      "derived-or-normalized"
      if not full_history_present
      else "present"
      if present
      else "absent"
    )
    cases.append(
      {
        "question_id": context.question_id,
        "question_type": reference.question_type,
        "abstention": reference.abstention,
        "context_sha256": context.context_sha256,
        "operand_sha256": _text_digest(reference.answer),
        "operand_present": present,
        "full_history_operand_present": full_history_present,
        "literal_status": status,
      }
    )
  answerable = [case for case in cases if not case["abstention"]]
  applicable = [case for case in answerable if case["full_history_operand_present"]]
  return _receipt(
    "agos-memory-lab-operand-probe-v2",
    sources={
      "contexts": {"file": args.contexts.name, "sha256": contexts_sha256},
      "references": source,
    },
    summary={
      "cases": len(cases),
      "answerable": len(answerable),
      "abstention": len(cases) - len(answerable),
      "literal_applicable": len(applicable),
      "derived_or_normalized": len(answerable) - len(applicable),
      "literal_present": sum(case["literal_status"] == "present" for case in applicable),
      "literal_absent": sum(case["literal_status"] == "absent" for case in applicable),
    },
    cases=cases,
  )


def _answerability(args: argparse.Namespace) -> dict[str, Any]:
  source = _longmem_receipt(args.receipt)
  if source.get("config", {}).get("retriever") != "lexical" or "source" in source.get("config", {}):
    raise ProbeError("answerability_requires_session_lexical_receipt")
  _, reference_source = qa._references(args)
  _matching_dataset(source, reference_source, error="answerability_dataset_mismatch")
  corpus = {case.question_id: case for case in _corpus(args)}
  candidates = source["config"].get("candidates")
  top_k = source["config"].get("top_k")
  if not isinstance(candidates, int) or candidates < 1 or not isinstance(top_k, int) or top_k < 1:
    raise ProbeError("answerability_receipt_invalid")
  cases = []
  for value in source["cases"]:
    case = corpus.get(value.get("question_id"))
    if case is None:
      raise ProbeError("answerability_corpus_mismatch")
    scores = _retrieval_scores(case, candidates=candidates, top_k=top_k)
    cases.append(
      {
        "question_id": value["question_id"],
        "question_type": value["question_type"],
        "answerable": not value["abstention"],
        **scores,
      }
    )
  return _receipt(
    "agos-memory-lab-answerability-probe-v1",
    sources={
      "retrieval": {
        "file": args.receipt.name,
        "sha256": _file_digest(args.receipt),
        "run_id": source["run_id"],
      },
      "references": reference_source,
    },
    summary=_answerability_summary(cases),
    cases=cases,
  )


def _retrieval_scores(case: longmem.Case, *, candidates: int, top_k: int) -> dict[str, float | None]:
  entries = longmem._session_entries(case)
  if not entries:
    return {"top1_score": None, "top1_margin": None, "mean_top_k_score": None}
  bm25 = longmem.BM25Okapi([entry.text.split(" ") for entry in entries])
  raw = bm25.get_scores(case.question.split(" "))
  scores = [float(raw[index]) for index in raw.argsort()[::-1][:candidates]]
  if not scores:
    return {"top1_score": None, "top1_margin": None, "mean_top_k_score": None}
  return {
    "top1_score": scores[0],
    "top1_margin": scores[0] - scores[1] if len(scores) > 1 else None,
    "mean_top_k_score": fmean(scores[:top_k]),
  }


def _answerability_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
  types = sorted({case["question_type"] for case in cases})
  return {
    "cases": len(cases),
    "answerable": sum(case["answerable"] for case in cases),
    "abstention": sum(not case["answerable"] for case in cases),
    "auroc": _aurocs(cases),
    "by_type": {
      question_type: {
        "cases": len(group),
        "answerable": sum(case["answerable"] for case in group),
        "abstention": sum(not case["answerable"] for case in group),
        "auroc": _aurocs(group),
      }
      for question_type in types
      if (group := [case for case in cases if case["question_type"] == question_type])
    },
  }


def _aurocs(cases: list[dict[str, Any]]) -> dict[str, float | None]:
  return {
    name: _auroc([(case[name], case["answerable"]) for case in cases if case[name] is not None])
    for name in ("top1_score", "top1_margin", "mean_top_k_score")
  }


def _auroc(values: list[tuple[float, bool]]) -> float | None:
  positive = [score for score, label in values if label]
  negative = [score for score, label in values if not label]
  if not positive or not negative:
    return None
  wins = sum((left > right) + 0.5 * (left == right) for left in positive for right in negative)
  return round(wins / (len(positive) * len(negative)), 8)


def _code_failures(args: argparse.Namespace) -> dict[str, Any]:
  retrieval = _longmem_receipt(args.receipt)
  contexts, contexts_sha256 = qa._contexts(args.contexts)
  references, reference_source = qa._references(args)
  corpus = _corpus(args)
  hypotheses = _hypotheses(args.hypotheses)
  evaluations = _evaluations(args.evaluations)
  comparison = _evaluations(args.comparison_evaluations) if args.comparison_evaluations else {}
  records, artifact = _artifact_records(args)
  _matching_dataset(retrieval, reference_source, error="failure_dataset_mismatch")
  if artifact is not None and (
    artifact.dataset_repository != reference_source["repository"]
    or artifact.dataset_revision != reference_source["revision"]
    or artifact.dataset_sha256 != reference_source["sha256"]
    or artifact.dataset_size != reference_source["size"]
  ):
    raise ProbeError("failure_artifact_dataset_mismatch")

  retrieval_by_id = _by_id(retrieval["cases"], error="failure_retrieval_identity_invalid")
  contexts_by_id = {context.question_id: context for context in contexts}
  references_by_id = {reference.question_id: reference for reference in references}
  corpus_by_id = {case.question_id: case for case in corpus}
  wrong = [value for value in evaluations.values() if not value["official_label"]]
  required = set(value["question_id"] for value in wrong)
  for values in (retrieval_by_id, contexts_by_id, references_by_id, corpus_by_id, hypotheses):
    if not required <= set(values):
      raise ProbeError("failure_source_identity_mismatch")

  cases = []
  for evaluation in wrong:
    question_id = evaluation["question_id"]
    reference = references_by_id[question_id]
    label = _failure_label(
      reference=reference,
      retrieval=retrieval_by_id[question_id],
      context=contexts_by_id[question_id].context,
      hypothesis=hypotheses[question_id],
      comparison=comparison.get(question_id),
      corpus=corpus_by_id[question_id],
      records=records.get(question_id),
    )
    cases.append(
      {
        "question_id": question_id,
        "question_type": reference.question_type,
        "abstention": reference.abstention,
        "label": label,
      }
    )
  counts = {label: sum(case["label"] == label for case in cases) for label in _FAILURES}
  sources: dict[str, Any] = {
    "retrieval": {"file": args.receipt.name, "sha256": _file_digest(args.receipt), "run_id": retrieval["run_id"]},
    "contexts": {"file": args.contexts.name, "sha256": contexts_sha256},
    "hypotheses": {"file": args.hypotheses.name, "sha256": _file_digest(args.hypotheses)},
    "evaluations": {"file": args.evaluations.name, "sha256": _file_digest(args.evaluations)},
    "references": reference_source,
  }
  if args.comparison_evaluations:
    sources["comparison_evaluations"] = {
      "file": args.comparison_evaluations.name,
      "sha256": _file_digest(args.comparison_evaluations),
    }
  if args.artifact:
    sources["artifact"] = {"file": args.artifact.name, "sha256": args.artifact_sha256}
  return _receipt(
    "agos-memory-lab-failure-probe-v1",
    sources=sources,
    summary={"judged_wrong": len(cases), "labels": counts},
    cases=cases,
  )


def _failure_label(
  *,
  reference: qa.Reference,
  retrieval: dict[str, Any],
  context: str,
  hypothesis: str,
  comparison: dict[str, Any] | None,
  corpus: longmem.Case,
  records: dict[str, str] | None,
) -> str:
  if comparison is not None and comparison["official_label"]:
    return "judge-disagreement"
  if reference.abstention:
    return "failed-abstention"
  if _contains(context, reference.answer):
    if _abstains(hypothesis):
      return "over-abstention"
    if _contains(hypothesis, reference.answer):
      return "reader-hedge"
    return "reader-unsupported"
  all_text, retrieved_text = _evidence(retrieval, corpus=corpus, records=records)
  if not _contains_any(all_text, reference.answer):
    return "fact-not-extracted"
  if not _contains_any(retrieved_text, reference.answer):
    return "fact-not-retrieved"
  return "lost-in-packing"


def _evidence(
  retrieval: dict[str, Any],
  *,
  corpus: longmem.Case,
  records: dict[str, str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
  sessions = {session.source_id: session.content for session in corpus.sessions}
  if "retrieved_memory_ids" in retrieval:
    if records is None:
      raise ProbeError("failure_memory_artifact_required")
    all_text = tuple(records.values())
    retrieved = tuple(records[item] for item in retrieval["retrieved_memory_ids"] if item in records)
    episode_ids = retrieval.get("retrieved_episode_occurrence_ids", ())
    return all_text, retrieved + tuple(sessions[item] for item in episode_ids)
  retrieved_ids = retrieval.get("retrieved_occurrence_ids")
  if not isinstance(retrieved_ids, list):
    raise ProbeError("failure_retrieval_receipt_invalid")
  try:
    return tuple(sessions.values()), tuple(sessions[item] for item in retrieved_ids)
  except KeyError as exc:
    raise ProbeError("failure_retrieval_source_missing") from exc


def _artifact_records(
  args: argparse.Namespace,
) -> tuple[dict[str, dict[str, str]], memory.Artifact | None]:
  if args.artifact is None and args.artifact_sha256 is None:
    return {}, None
  if args.artifact is None or not args.artifact_sha256:
    raise ProbeError("failure_artifact_identity_required")
  artifact = memory.load(args.artifact, sha256=args.artifact_sha256)
  grouped: dict[str, dict[str, str]] = {}
  for record in artifact.records:
    grouped.setdefault(record.case_id, {})[record.record_id] = record.text
  return grouped, artifact


def _corpus(args: argparse.Namespace) -> tuple[longmem.Case, ...]:
  path = args.references or args.data / longmem._DATASETS[args.dataset or "s"]["file"]
  return longmem._load(path)


def _hypotheses(path: Path) -> dict[str, str]:
  values, _ = qa._hypotheses(path)
  return {question_id: value["hypothesis"] for question_id, value in values.items()}


def _evaluations(path: Path) -> dict[str, dict[str, Any]]:
  try:
    values = qa._jsonl(path, error="failure_evaluation_invalid")
  except qa.QAError as exc:
    raise ProbeError(str(exc)) from exc
  records: dict[str, dict[str, Any]] = {}
  for value in values:
    if not isinstance(value, dict) or not isinstance(value.get("question_id"), str) or not isinstance(value.get("official_label"), bool):
      raise ProbeError("failure_evaluation_invalid")
    if value["question_id"] in records:
      raise ProbeError("failure_evaluation_identity_duplicated")
    records[value["question_id"]] = value
  return records


def _longmem_receipt(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    raise ProbeError("retrieval_receipt_invalid") from exc
  if not isinstance(value, dict) or value.get("schema") != "agos-memory-lab-longmem-v1" or not isinstance(value.get("cases"), list):
    raise ProbeError("retrieval_receipt_invalid")
  receipt_sha256 = value.get("receipt_sha256")
  if receipt_sha256 is not None:
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if receipt_sha256 != _digest(body):
      raise ProbeError("retrieval_receipt_digest_mismatch")
  return value


def _matching_dataset(receipt: dict[str, Any], source: dict[str, Any], *, error: str) -> None:
  if receipt.get("dataset") != source:
    raise ProbeError(error)


def _by_id(values: list[dict[str, Any]], *, error: str) -> dict[str, dict[str, Any]]:
  result: dict[str, dict[str, Any]] = {}
  for value in values:
    question_id = value.get("question_id") if isinstance(value, dict) else None
    if not isinstance(question_id, str) or question_id in result:
      raise ProbeError(error)
    result[question_id] = value
  return result


def _abstains(value: str) -> bool:
  return bool(
    re.search(
      r"\b(unanswerable|insufficient|cannot answer|can't answer|not enough (?:information|evidence)|history does not)\b",
      value,
      re.IGNORECASE,
    )
  )


def _contains_any(values: tuple[str, ...], operand: str) -> bool:
  return any(_contains(value, operand) for value in values)


def _contains(value: str, operand: str) -> bool:
  return operand.casefold() in value.casefold()


def _receipt(schema: str, *, sources: dict[str, Any], summary: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
  semantic = {"schema": schema, "sources": sources, "summary": summary, "cases": cases}
  receipt = {**semantic, "run_id": _digest(semantic)}
  return {**receipt, "receipt_sha256": _digest(receipt)}


def _write(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix(f"{path.suffix}.part")
  partial.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
  os.replace(partial, path)


def _text_digest(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _digest(value: Any) -> str:
  return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
  main()
