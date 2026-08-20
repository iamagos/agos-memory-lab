from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_LONGMEM_SCHEMA = "agos-memory-lab-longmem-v1"
_READ_SCHEMA = "agos-memory-lab-read-v4"
_JUDGE_SCHEMA = "agos-memory-lab-judge-v4"
_CONTROL_KEYS = (
  "retriever",
  "offset",
  "limit",
  "candidates",
  "top_k",
  "chars",
  "lexical_weight",
)


@dataclass(frozen=True, slots=True)
class Receipt:
  path: Path
  sha256: str
  value: dict[str, Any]

  @property
  def identity(self) -> dict[str, str]:
    return {
      "sha256": self.sha256,
      "receipt_sha256": self.value["receipt_sha256"],
      "run_id": self.value["run_id"],
    }


class CompareError(Exception):
  pass


def main() -> None:
  args = _parser().parse_args()
  try:
    receipt = _compare(args)
    _write(args.out, receipt)
    print(
      json.dumps(
        {
          "comparison_id": receipt["comparison_id"],
          "out": str(args.out),
          "sessions": receipt["sessions"]["longmem"]["run_id"],
          "memories": receipt["memories"]["longmem"]["run_id"],
        },
        indent=2,
        sort_keys=True,
      )
    )
  except CompareError as exc:
    raise SystemExit(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Compare exact LongMemEval session and memory receipts.")
  parser.add_argument("--sessions", type=Path, required=True, help="Raw-session LongMemEval receipt.")
  parser.add_argument("--memories", type=Path, required=True, help="Memory LongMemEval receipt.")
  parser.add_argument("--sessions-read", type=Path)
  parser.add_argument("--memories-read", type=Path)
  parser.add_argument("--sessions-judge", type=Path)
  parser.add_argument("--memories-judge", type=Path)
  parser.add_argument("--out", type=Path, required=True)
  return parser


def _compare(args: argparse.Namespace) -> dict[str, Any]:
  _distinct(args)
  _pairs(args)
  sessions = _longmem(_load(args.sessions, error="sessions_receipt_invalid"), source="sessions")
  memories = _longmem(_load(args.memories, error="memories_receipt_invalid"), source="memories")
  contract = _contract(sessions.value, memories.value)

  session_read = _optional(args.sessions_read, kind="read")
  memory_read = _optional(args.memories_read, kind="read")
  session_judge = _optional(args.sessions_judge, kind="judge")
  memory_judge = _optional(args.memories_judge, kind="judge")
  if session_read is not None and memory_read is not None:
    _bind_read(sessions.value, session_read.value)
    _bind_read(memories.value, memory_read.value)
    if _qa_contract(session_read.value) != _qa_contract(memory_read.value):
      raise CompareError("comparison_reader_contract_mismatch")
    contract["reader"] = _qa_contract(session_read.value)
  else:
    contract["reader"] = None
  if session_judge is not None and memory_judge is not None:
    _bind_judge(sessions.value, session_read.value, session_judge.value)
    _bind_judge(memories.value, memory_read.value, memory_judge.value)
    if _qa_contract(session_judge.value) != _qa_contract(memory_judge.value):
      raise CompareError("comparison_judge_contract_mismatch")
    contract["judge"] = _qa_contract(session_judge.value)
  else:
    contract["judge"] = None

  session_report = _report(sessions, read=session_read, judge=session_judge)
  memory_report = _report(memories, read=memory_read, judge=memory_judge)
  semantic = {
    "schema": "agos-memory-lab-comparison-v2",
    "contract": contract,
    "sessions": session_report,
    "memories": memory_report,
    "delta": _delta(session_report, memory_report),
  }
  comparison_id = _digest(semantic)
  receipt = {**semantic, "comparison_id": comparison_id}
  return {**receipt, "receipt_sha256": _digest(receipt)}


def _pairs(args: argparse.Namespace) -> None:
  reads = (args.sessions_read, args.memories_read)
  judges = (args.sessions_judge, args.memories_judge)
  if any(value is not None for value in reads) and not all(value is not None for value in reads):
    raise CompareError("comparison_reader_pair_required")
  if any(value is not None for value in judges) and (
    not all(value is not None for value in judges)
    or not all(value is not None for value in reads)
  ):
    raise CompareError("comparison_judge_pair_required")


def _distinct(args: argparse.Namespace) -> None:
  paths = tuple(
    path.resolve()
    for path in (
      args.sessions,
      args.memories,
      args.sessions_read,
      args.memories_read,
      args.sessions_judge,
      args.memories_judge,
      args.out,
    )
    if path is not None
  )
  if len(paths) != len(set(paths)):
    raise CompareError("comparison_path_conflict")


def _optional(path: Path | None, *, kind: str) -> Receipt | None:
  if path is None:
    return None
  return _qa(_load(path, error=f"{kind}_receipt_invalid"), kind=kind)


def _load(path: Path, *, error: str) -> Receipt:
  try:
    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=_invalid_constant)
  except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
    raise CompareError(error) from exc
  if not isinstance(value, dict):
    raise CompareError(error)
  receipt_sha256 = _sha256(value.get("receipt_sha256"), error=error)
  body = {key: item for key, item in value.items() if key != "receipt_sha256"}
  if receipt_sha256 != _digest(body):
    raise CompareError(f"{error}_digest_mismatch")
  return Receipt(path, hashlib.sha256(raw).hexdigest(), value)


def _longmem(receipt: Receipt, *, source: str) -> Receipt:
  value = receipt.value
  expected = {
    "schema",
    "benchmark",
    "dataset",
    "kernel",
    "config",
    "summary",
    "cases",
    "run_id",
    "measurements",
    "contexts",
    "receipt_sha256",
  }
  if set(value) != expected or value.get("schema") != _LONGMEM_SCHEMA:
    raise CompareError(f"{source}_receipt_shape_invalid")
  semantic = {
    key: value[key]
    for key in expected - {"run_id", "measurements", "contexts", "receipt_sha256"}
  }
  if _sha256(value["run_id"], error=f"{source}_run_identity_invalid") != _digest(semantic):
    raise CompareError(f"{source}_run_identity_mismatch")
  config = _object(value["config"], error=f"{source}_config_invalid")
  expected_source = config.get("source", "sessions")
  if expected_source != source or (source == "sessions" and "artifact" in config):
    raise CompareError(f"{source}_source_invalid")
  if source == "memories" and not isinstance(config.get("artifact"), dict):
    raise CompareError("memories_artifact_missing")
  _case_ids(value, error=f"{source}_cases_invalid")
  _measurements(value["measurements"], error=f"{source}_measurements_invalid")
  if value["contexts"] is not None:
    contexts = _object(value["contexts"], error=f"{source}_contexts_invalid")
    if set(contexts) != {"file", "sha256"}:
      raise CompareError(f"{source}_contexts_invalid")
    _text(contexts["file"], error=f"{source}_contexts_invalid")
    _sha256(contexts["sha256"], error=f"{source}_contexts_invalid")
  _longmem_summary(value, error=f"{source}_summary_invalid")
  return receipt


def _qa(receipt: Receipt, *, kind: str) -> Receipt:
  value = receipt.value
  schema = _READ_SCHEMA if kind == "read" else _JUDGE_SCHEMA
  semantic_keys = {
    "schema",
    "benchmark_revision",
    "source",
    "config",
    "prompt_revision",
    "cases",
  }
  if kind == "judge":
    semantic_keys.add("scores")
  expected = semantic_keys | {"run_id", "summary", "output", "receipt_sha256"}
  if set(value) != expected or value.get("schema") != schema:
    raise CompareError(f"{kind}_receipt_shape_invalid")
  semantic = {key: value[key] for key in semantic_keys}
  if _sha256(value["run_id"], error=f"{kind}_run_identity_invalid") != _digest(semantic):
    raise CompareError(f"{kind}_run_identity_mismatch")
  _case_ids(value, error=f"{kind}_cases_invalid")
  _object(value["config"], error=f"{kind}_config_invalid")
  if kind == "judge":
    _object(value["scores"], error="judge_scores_invalid")
  _qa_summary(value, kind=kind)
  output = _object(value["output"], error=f"{kind}_output_invalid")
  _text(output.get("file"), error=f"{kind}_output_invalid")
  _sha256(output.get("sha256"), error=f"{kind}_output_invalid")
  return receipt


def _contract(sessions: dict[str, Any], memories: dict[str, Any]) -> dict[str, Any]:
  for field in ("benchmark", "dataset", "kernel"):
    if sessions[field] != memories[field]:
      raise CompareError(f"comparison_{field}_mismatch")
  session_config = _retrieval_contract(sessions["config"], source="sessions")
  memory_config = _retrieval_contract(memories["config"], source="memories")
  episodes = memory_config.pop("episodes", 0)
  if session_config != memory_config:
    raise CompareError("comparison_retrieval_contract_mismatch")
  cases = _case_ids(sessions, error="sessions_cases_invalid")
  if cases != _case_ids(memories, error="memories_cases_invalid"):
    raise CompareError("comparison_case_window_mismatch")
  contract = {
    "benchmark": sessions["benchmark"],
    "dataset": sessions["dataset"],
    "kernel": sessions["kernel"],
    "cases": cases,
    "retrieval": session_config,
  }
  if episodes:
    contract["treatment"] = {"episode_candidates": episodes}
  return contract


def _retrieval_contract(value: Any, *, source: str) -> dict[str, Any]:
  config = _object(value, error=f"{source}_config_invalid")
  expected = set(_CONTROL_KEYS) | {"retriever_identity"}
  if source == "memories":
    expected |= {"source", "artifact"}
    if "episodes" in config:
      expected.add("episodes")
  if set(config) != expected:
    raise CompareError(f"{source}_config_invalid")
  episodes = config.get("episodes", 0)
  top_k = config.get("top_k")
  if (
    not isinstance(episodes, int)
    or isinstance(episodes, bool)
    or not isinstance(top_k, int)
    or isinstance(top_k, bool)
    or not 0 <= episodes <= top_k
  ):
    raise CompareError("memories_episode_candidates_invalid")
  identity = config["retriever_identity"]
  if episodes:
    identity = _object(identity, error="memories_retriever_identity_invalid")
    if set(identity) != {"memories", "episodes"}:
      raise CompareError("memories_retriever_identity_invalid")
    memory_identity = _without_text(identity["memories"], expected="memory text")
    episode_identity = _without_text(identity["episodes"], expected="user turns")
    if memory_identity != episode_identity:
      raise CompareError("comparison_retrieval_contract_mismatch")
  else:
    memory_identity = _without_text(
      identity,
      expected="user turns" if source == "sessions" else "memory text",
    )
  contract = {
    **{key: config[key] for key in _CONTROL_KEYS},
    "retriever_identity": memory_identity,
  }
  if episodes:
    contract["episodes"] = episodes
  return contract


def _qa_contract(value: dict[str, Any]) -> dict[str, Any]:
  config = value["config"]
  return {
    "benchmark_revision": value["benchmark_revision"],
    "config": {
      "request": config["request"],
      "execution": {
        key: item
        for key, item in config["execution"].items()
        if key != "max_cost_usd"
      },
    },
    "prompt_revision": value["prompt_revision"],
  }


def _bind_read(longmem: dict[str, Any], read: dict[str, Any]) -> None:
  source = _object(read["source"], error="comparison_reader_source_invalid")
  contexts = _object(longmem["contexts"], error="comparison_reader_contexts_missing")
  if (
    source.get("run_id") != longmem["run_id"]
    or source.get("sha256") != contexts.get("sha256")
    or read["benchmark_revision"] != longmem["benchmark"]["revision"]
    or _case_ids(read, error="read_cases_invalid") != _case_ids(longmem, error="longmem_cases_invalid")
  ):
    raise CompareError("comparison_reader_source_mismatch")


def _bind_judge(longmem: dict[str, Any], read: dict[str, Any], judge: dict[str, Any]) -> None:
  source = _object(judge["source"], error="comparison_judge_source_invalid")
  hypotheses = _object(source.get("hypotheses"), error="comparison_judge_source_invalid")
  if (
    hypotheses.get("sha256") != read["output"]["sha256"]
    or source.get("references") != longmem["dataset"]
    or judge["benchmark_revision"] != longmem["benchmark"]["revision"]
    or _case_ids(judge, error="judge_cases_invalid") != _case_ids(longmem, error="longmem_cases_invalid")
  ):
    raise CompareError("comparison_judge_source_mismatch")


def _report(
  longmem: Receipt,
  *,
  read: Receipt | None,
  judge: Receipt | None,
) -> dict[str, Any]:
  value = longmem.value
  config = value["config"]
  return {
    "longmem": {
      **longmem.identity,
      "retrieval": value["summary"]["retrieval"],
      "governed": value["summary"]["kernel"],
      "context": value["summary"]["selection"],
      "measurements": value["measurements"],
      "contexts": value["contexts"],
      "support": _support(value),
    },
    "artifact": config.get("artifact"),
    "reader": _qa_report(read),
    "judge": _qa_report(judge),
    "economics": _economics(read, judge),
  }


def _qa_report(receipt: Receipt | None) -> dict[str, Any] | None:
  if receipt is None:
    return None
  return {**receipt.identity, "summary": receipt.value["summary"]}


def _economics(read: Receipt | None, judge: Receipt | None) -> dict[str, Any] | None:
  if read is None or judge is None:
    return None
  summaries = (read.value["summary"], judge.value["summary"])
  estimated = tuple(summary.get("estimated_cost_usd") for summary in summaries)
  return {
    "calls": sum(_count(summary.get("cases"), error="qa_case_count_invalid") for summary in summaries),
    "usage_missing": sum(_count(summary.get("usage_missing"), error="qa_usage_invalid") for summary in summaries),
    "input_tokens": sum(_count(summary.get("input_tokens"), error="qa_token_count_invalid") for summary in summaries),
    "output_tokens": sum(_count(summary.get("output_tokens"), error="qa_token_count_invalid") for summary in summaries),
    "latency_seconds": round(sum(_number(summary.get("latency_seconds"), error="qa_latency_invalid") for summary in summaries), 6),
    "estimated_cost_usd": (
      round(sum(_number(value, error="qa_cost_invalid") for value in estimated), 8)
      if all(value is not None for value in estimated)
      else None
    ),
    "reserved_cost_usd": round(
      sum(_number(summary.get("reserved_cost_usd"), error="qa_cost_invalid") for summary in summaries),
      8,
    ),
  }


def _delta(sessions: dict[str, Any], memories: dict[str, Any]) -> dict[str, Any]:
  session_longmem = sessions["longmem"]
  memory_longmem = memories["longmem"]
  session_economics = sessions["economics"]
  memory_economics = memories["economics"]
  return {
    "direction": "memories_minus_sessions",
    "retrieval": _metric_delta(session_longmem["retrieval"], memory_longmem["retrieval"]),
    "governed": _metric_delta(session_longmem["governed"], memory_longmem["governed"]),
    "mean_context_chars": _difference(
      session_longmem["context"].get("mean_content_chars"),
      memory_longmem["context"].get("mean_content_chars"),
    ),
    "retrieval_seconds": _difference(
      session_longmem["measurements"].get("total_seconds"),
      memory_longmem["measurements"].get("total_seconds"),
    ),
    "qa_accuracy": _difference(
      _nested(sessions, "judge", "summary", "accuracy"),
      _nested(memories, "judge", "summary", "accuracy"),
    ),
    "model_calls": _difference(
      session_economics.get("calls") if session_economics is not None else None,
      memory_economics.get("calls") if memory_economics is not None else None,
    ),
    "model_tokens": _difference(
      _total_tokens(session_economics),
      _total_tokens(memory_economics),
    ),
    "estimated_cost_usd": _difference(
      session_economics.get("estimated_cost_usd") if session_economics is not None else None,
      memory_economics.get("estimated_cost_usd") if memory_economics is not None else None,
    ),
  }


def _metric_delta(sessions: Any, memories: Any) -> dict[str, float] | None:
  if not isinstance(sessions, dict) or not isinstance(memories, dict) or set(sessions) != set(memories):
    return None
  result = {}
  for key in sessions:
    value = _difference(sessions[key], memories[key])
    if value is None:
      return None
    result[key] = value
  return result


def _difference(sessions: Any, memories: Any) -> float | int | None:
  if not _is_number(sessions) or not _is_number(memories):
    return None
  value = memories - sessions
  return round(value, 8) if isinstance(value, float) else value


def _total_tokens(value: dict[str, Any] | None) -> int | None:
  if value is None or value["usage_missing"]:
    return None
  return value["input_tokens"] + value["output_tokens"]


def _nested(value: dict[str, Any], *keys: str) -> Any:
  current: Any = value
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return current


def _case_ids(value: dict[str, Any], *, error: str) -> list[str]:
  cases = value.get("cases")
  if not isinstance(cases, list):
    raise CompareError(error)
  result = []
  for case in cases:
    if not isinstance(case, dict):
      raise CompareError(error)
    result.append(_text(case.get("question_id"), error=error))
  if not result or len(result) != len(set(result)):
    raise CompareError(error)
  return result


def _longmem_summary(value: dict[str, Any], *, error: str) -> None:
  summary = _object(value["summary"], error=error)
  expected = {
    "cases",
    "eligible",
    "ignored_abstention",
    "ignored_missing_target",
    "retrieval",
    "kernel",
    "selection",
    "by_type",
  }
  if set(summary) != expected or _count(summary["cases"], error=error) != len(value["cases"]):
    raise CompareError(error)
  for key in ("eligible", "ignored_abstention", "ignored_missing_target"):
    _count(summary[key], error=error)
  _metrics(summary["retrieval"], error=error)
  _metrics(summary["kernel"], error=error)
  selection = _object(summary["selection"], error=error)
  if set(selection) != {
    "truncated_cases",
    "mean_candidates",
    "mean_selected",
    "mean_content_chars",
    "outcomes",
  }:
    raise CompareError(error)
  _count(selection["truncated_cases"], error=error)
  for key in ("mean_candidates", "mean_selected", "mean_content_chars"):
    _number(selection[key], error=error)
  outcomes = _object(selection["outcomes"], error=error)
  for count in outcomes.values():
    _count(count, error=error)


def _qa_summary(value: dict[str, Any], *, kind: str) -> None:
  summary = _object(value["summary"], error=f"{kind}_summary_invalid")
  required = {
    "cases",
    "input_tokens",
    "output_tokens",
    "usage_missing",
    "estimated_cost_usd",
    "reserved_cost_usd",
    "latency_seconds",
  }
  if not required <= set(summary) or _count(summary["cases"], error=f"{kind}_summary_invalid") != len(value["cases"]):
    raise CompareError(f"{kind}_summary_invalid")
  for key in ("input_tokens", "output_tokens", "usage_missing"):
    _count(summary[key], error=f"{kind}_summary_invalid")
  if summary["estimated_cost_usd"] is not None:
    _number(summary["estimated_cost_usd"], error=f"{kind}_summary_invalid")
  for key in ("reserved_cost_usd", "latency_seconds"):
    _number(summary[key], error=f"{kind}_summary_invalid")
  if kind == "judge" and summary.get("accuracy") != value["scores"].get("accuracy"):
    raise CompareError("judge_summary_invalid")


def _metrics(value: Any, *, error: str) -> None:
  metrics = _object(value, error=error)
  for score in metrics.values():
    _number(score, error=error)


def _support(value: dict[str, Any]) -> dict[str, Any] | None:
  if value["config"].get("source", "sessions") == "sessions":
    return None
  counts = {"current": 0, "missing": 0, "replaced": 0, "stale": 0}
  for case in value["cases"]:
    kernel = _object(case.get("kernel"), error="memory_support_invalid")
    decisions = kernel.get("support")
    if not isinstance(decisions, list):
      raise CompareError("memory_support_invalid")
    for decision in decisions:
      if not isinstance(decision, dict) or decision.get("decision") not in counts:
        raise CompareError("memory_support_invalid")
      for key in ("record_id", "source_ref", "source_occurrence_id", "source_digest", "reopened_digest"):
        _text(decision.get(key), error="memory_support_invalid")
      counts[decision["decision"]] += 1
  return {
    "checked": sum(counts.values()),
    "failures": sum(count for decision, count in counts.items() if decision != "current"),
    "decisions": counts,
  }


def _measurements(value: Any, *, error: str) -> None:
  expected = {
    "load_seconds",
    "index_seconds",
    "query_seconds",
    "kernel_seconds",
    "total_seconds",
  }
  if not isinstance(value, dict) or set(value) != expected:
    raise CompareError(error)
  for item in value.values():
    _number(item, error=error)


def _without_text(value: Any, *, expected: str) -> Any:
  if isinstance(value, dict):
    if "text" in value and value["text"] != expected:
      raise CompareError("comparison_retrieval_text_invalid")
    return {
      key: _without_text(item, expected=expected)
      for key, item in value.items()
      if key != "text"
    }
  if isinstance(value, list):
    return [_without_text(item, expected=expected) for item in value]
  return value


def _object(value: Any, *, error: str) -> dict[str, Any]:
  if not isinstance(value, dict):
    raise CompareError(error)
  return value


def _text(value: Any, *, error: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise CompareError(error)
  return value


def _sha256(value: Any, *, error: str) -> str:
  if (
    not isinstance(value, str)
    or len(value) != 64
    or any(character not in "0123456789abcdef" for character in value)
  ):
    raise CompareError(error)
  return value


def _count(value: Any, *, error: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise CompareError(error)
  return value


def _number(value: Any, *, error: str) -> float:
  if not _is_number(value) or not math.isfinite(value) or value < 0:
    raise CompareError(error)
  return float(value)


def _is_number(value: Any) -> bool:
  return isinstance(value, int | float) and not isinstance(value, bool)


def _invalid_constant(_: str) -> None:
  raise ValueError


def _digest(value: Any) -> str:
  try:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
  except (TypeError, ValueError) as exc:
    raise CompareError("comparison_value_invalid") from exc
  return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix(f"{path.suffix}.part")
  partial.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
  os.replace(partial, path)


if __name__ == "__main__":
  main()
