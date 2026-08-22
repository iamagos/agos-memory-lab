from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import longmem
import model as chat_model
import call as bounded
from call import Config as ChatConfig


_PROMPT_REVISION = "longmem-direct-v1"
_JUDGE_REVISION = f"longmem-official-{longmem._BENCHMARK_REVISION}"
_CHAT_ENVELOPE_TOKENS = 256
_TASKS = (
  "single-session-user",
  "single-session-assistant",
  "single-session-preference",
  "temporal-reasoning",
  "knowledge-update",
  "multi-session",
)


@dataclass(frozen=True, slots=True)
class Context:
  run_id: str
  question_id: str
  question: str
  question_date: str
  context: str
  context_sha256: str


@dataclass(frozen=True, slots=True)
class Reference:
  question_id: str
  question_type: str
  question: str
  answer: str
  abstention: bool


ChatResult = chat_model.ModelResult


class QAError(Exception):
  pass


def main() -> None:
  parser = _parser()
  args = parser.parse_args()
  try:
    if args.command == "read":
      _read(args)
    elif args.command == "judge":
      _judge(args)
    else:
      parser.error("command_required")
  except QAError as exc:
    raise SystemExit(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Read and judge governed LongMemEval contexts.")
  commands = parser.add_subparsers(dest="command")

  read = commands.add_parser("read", help="Generate official hypothesis records.")
  read.add_argument("--contexts", type=Path, required=True)
  read.add_argument("--out", type=Path, required=True)
  read.add_argument("--receipt", type=Path)
  read.add_argument("--offset", type=int, default=0)
  read.add_argument("--limit", type=int, default=0)
  _chat_arguments(read, default_tokens=1_000)

  judge = commands.add_parser("judge", help="Apply the pinned official QA judge.")
  judge.add_argument("--hypotheses", type=Path, required=True)
  source = judge.add_mutually_exclusive_group()
  source.add_argument("--dataset", choices=tuple(longmem._DATASETS), default="s")
  source.add_argument("--references", type=Path)
  judge.add_argument("--sha256")
  judge.add_argument("--revision")
  judge.add_argument("--data", type=Path, default=Path("data"))
  judge.add_argument("--out", type=Path, required=True)
  judge.add_argument("--receipt", type=Path)
  judge.add_argument("--offset", type=int, default=0)
  judge.add_argument("--limit", type=int, default=0)
  _chat_arguments(judge, default_tokens=10)
  return parser


def _chat_arguments(parser: argparse.ArgumentParser, *, default_tokens: int) -> None:
  bounded.arguments(parser, default_tokens=default_tokens)


def _read(args: argparse.Namespace) -> None:
  config = _config(args)
  receipt_path = args.receipt or _receipt_path(args.out)
  pending_path = _pending_path(args.out)
  _distinct_paths(args.contexts, args.out, receipt_path, pending_path)
  contexts, contexts_sha256 = _contexts(args.contexts)
  contexts = _window(contexts, offset=args.offset, limit=args.limit)
  existing = _existing(args.out, kind="hypothesis")
  expected = {context.question_id: _reader_id(context, config) for context in contexts}
  _validate_existing(existing, expected, error="reader_resume_identity_mismatch")
  for context in contexts:
    if record := existing.get(context.question_id):
      _validate_record_cost(record, prompt=_reader_prompt(context), config=config, error="hypothesis_resume_invalid")
  _recover_pending(pending_path, records=existing)

  records = dict(existing)
  key: str | None = None
  for context in contexts:
    if context.question_id in records:
      continue
    prompt = _reader_prompt(context)
    _check_cost(records.values(), prompt=prompt, config=config)
    if key is None:
      key = _key(args, config=config)
    _begin_pending(pending_path, question_id=context.question_id, request_id=expected[context.question_id], prompt=prompt)
    started = time.perf_counter()
    result = _chat(prompt, config=config, key=key)
    cost = _cost(result, prompt=prompt, config=config)
    record = {
      "schema": "agos-memory-lab-hypothesis-v1",
      "context_run_id": context.run_id,
      "context_sha256": context.context_sha256,
      "question_id": context.question_id,
      "request_id": expected[context.question_id],
      "hypothesis": result.content,
      "hypothesis_sha256": _text_digest(result.content),
      "reader": {"requested_model": config.model, "response_model": result.model},
      "usage": _usage(result),
      "cost": cost,
      "latency_seconds": round(time.perf_counter() - started, 6),
    }
    records[context.question_id] = record
    _write_jsonl(args.out, [records[item.question_id] for item in contexts if item.question_id in records])
    _clear_pending(pending_path)

  ordered = [records[context.question_id] for context in contexts]
  receipt = _run_receipt(
    kind="read",
    source={
      "path": args.contexts.name,
      "sha256": contexts_sha256,
      "run_id": _one(context.run_id for context in contexts),
    },
    config=config,
    records=ordered,
    output=args.out,
  )
  _write(receipt_path, receipt)
  print(json.dumps({"out": str(args.out), "receipt": str(receipt_path), "summary": receipt["summary"]}, sort_keys=True))


def _judge(args: argparse.Namespace) -> None:
  config = _config(args)
  receipt_path = args.receipt or _receipt_path(args.out)
  pending_path = _pending_path(args.out)
  reference_path = args.references or args.data / longmem._DATASETS[args.dataset or "s"]["file"]
  _distinct_paths(args.hypotheses, reference_path, args.out, receipt_path, pending_path)
  hypotheses, hypotheses_sha256 = _hypotheses(args.hypotheses)
  hypotheses = _mapping_window(hypotheses, offset=args.offset, limit=args.limit)
  references, source = _references(args)
  by_id = {reference.question_id: reference for reference in references}
  if any(question_id not in by_id for question_id in hypotheses):
    raise QAError("judge_hypothesis_unknown")

  existing = _existing(args.out, kind="evaluation")
  expected = {
    question_id: _judge_id(by_id[question_id], hypothesis["hypothesis"], config)
    for question_id, hypothesis in hypotheses.items()
  }
  _validate_existing(existing, expected, error="judge_resume_identity_mismatch")
  for question_id, hypothesis in hypotheses.items():
    if record := existing.get(question_id):
      prompt = _judge_prompt(by_id[question_id], hypothesis["hypothesis"])
      _validate_record_cost(record, prompt=prompt, config=config, error="evaluation_resume_invalid")
  _recover_pending(pending_path, records=existing)

  records = dict(existing)
  key: str | None = None
  for question_id, hypothesis in hypotheses.items():
    if question_id in records:
      continue
    reference = by_id[question_id]
    prompt = _judge_prompt(reference, hypothesis["hypothesis"])
    _check_cost(records.values(), prompt=prompt, config=config)
    if key is None:
      key = _key(args, config=config)
    _begin_pending(pending_path, question_id=question_id, request_id=expected[question_id], prompt=prompt)
    started = time.perf_counter()
    result = _chat(prompt, config=config, key=key)
    official, strict = _labels(result.content)
    cost = _cost(result, prompt=prompt, config=config)
    record = {
      "schema": "agos-memory-lab-evaluation-v1",
      "question_id": question_id,
      "question_type": reference.question_type,
      "abstention": reference.abstention,
      "request_id": expected[question_id],
      "hypothesis_sha256": _text_digest(hypothesis["hypothesis"]),
      "official_label": official,
      "strict_label": strict,
      "judge_response": result.content,
      "judge": {"requested_model": config.model, "response_model": result.model},
      "usage": _usage(result),
      "cost": cost,
      "latency_seconds": round(time.perf_counter() - started, 6),
    }
    records[question_id] = record
    _write_jsonl(args.out, [records[item] for item in hypotheses if item in records])
    _clear_pending(pending_path)

  ordered = [records[question_id] for question_id in hypotheses]
  receipt = _run_receipt(
    kind="judge",
    source={
      "hypotheses": {"path": args.hypotheses.name, "sha256": hypotheses_sha256},
      "references": source,
    },
    config=config,
    records=ordered,
    output=args.out,
    scores=_scores(ordered),
  )
  _write(receipt_path, receipt)
  print(json.dumps({"out": str(args.out), "receipt": str(receipt_path), "summary": receipt["summary"]}, sort_keys=True))


def _config(args: argparse.Namespace) -> ChatConfig:
  try:
    return bounded.config(args)
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc


def _key(args: argparse.Namespace, *, config: ChatConfig) -> str:
  try:
    return bounded.key(args, config=config)
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc


def _chat(prompt: str, *, config: ChatConfig, key: str) -> ChatResult:
  try:
    return chat_model.complete(prompt, config=config, api_key=key)
  except chat_model.ModelError as exc:
    raise QAError(str(exc)) from exc


def _reader_prompt(context: Context) -> str:
  return (
    "I will give you several history chats between you and a user. Please answer the question based on the relevant "
    "chat history. Answer the question step by step: first extract all the relevant information, and then reason over "
    "the information to get the answer.\n\n\n"
    f"History Chats:\n\n{context.context}\n\nCurrent Date: {context.question_date}\n"
    f"Question: {context.question}\nAnswer (step by step):"
  )


def _judge_prompt(reference: Reference, hypothesis: str) -> str:
  if reference.abstention:
    return (
      "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if "
      "the model correctly identifies the question as unanswerable. The model could say that the information is "
      "incomplete, or some other information is given but the asked information is not."
      f"\n\nQuestion: {reference.question}\n\nExplanation: {reference.answer}\n\nModel Response: {hypothesis}"
      "\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    )
  if reference.question_type == "single-session-preference":
    return (
      "I will give you a question, a rubric for desired personalized response, and a response from a model. Please "
      "answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to "
      "reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's "
      "personal information correctly."
      f"\n\nQuestion: {reference.question}\n\nRubric: {reference.answer}\n\nModel Response: {hypothesis}"
      "\n\nIs the model response correct? Answer yes or no only."
    )
  if reference.question_type == "temporal-reasoning":
    return (
      "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
      "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
      "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
      "only contains a subset of the information required by the answer, answer no. In addition, do not penalize "
      "off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and "
      "the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is "
      "still correct."
      f"\n\nQuestion: {reference.question}\n\nCorrect Answer: {reference.answer}\n\nModel Response: {hypothesis}"
      "\n\nIs the model response correct? Answer yes or no only."
    )
  if reference.question_type == "knowledge-update":
    return (
      "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
      "contains the correct answer. Otherwise, answer no. If the response contains some previous information along "
      "with an updated answer, the response should be considered as correct as long as the updated answer is the "
      "required answer."
      f"\n\nQuestion: {reference.question}\n\nCorrect Answer: {reference.answer}\n\nModel Response: {hypothesis}"
      "\n\nIs the model response correct? Answer yes or no only."
    )
  if reference.question_type in {"single-session-user", "single-session-assistant", "multi-session"}:
    return (
      "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
      "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
      "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
      "only contains a subset of the information required by the answer, answer no."
      f"\n\nQuestion: {reference.question}\n\nCorrect Answer: {reference.answer}\n\nModel Response: {hypothesis}"
      "\n\nIs the model response correct? Answer yes or no only."
    )
  raise QAError("judge_question_type_invalid")


def _contexts(path: Path) -> tuple[tuple[Context, ...], str]:
  values = _jsonl(path, error="context_file_invalid")
  contexts = tuple(_context(value) for value in values)
  _unique((context.question_id for context in contexts), error="context_question_identity_duplicated")
  _one(context.run_id for context in contexts)
  return contexts, _file_digest(path)


def _context(value: Any) -> Context:
  if not isinstance(value, dict):
    raise QAError("context_record_invalid")
  context = _string(value.get("context"), error="context_text_invalid", empty=True)
  digest = _digest_string(value.get("context_sha256"), error="context_digest_invalid")
  if _text_digest(context) != digest:
    raise QAError("context_digest_mismatch")
  return Context(
    run_id=_digest_string(value.get("run_id"), error="context_run_identity_invalid"),
    question_id=_string(value.get("question_id"), error="context_question_identity_invalid"),
    question=_string(value.get("question"), error="context_question_invalid"),
    question_date=_string(value.get("question_date"), error="context_question_date_invalid"),
    context=context,
    context_sha256=digest,
  )


def _hypotheses(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
  values = _jsonl(path, error="hypothesis_file_invalid")
  records: dict[str, dict[str, Any]] = {}
  for value in values:
    if not isinstance(value, dict):
      raise QAError("hypothesis_record_invalid")
    question_id = _string(value.get("question_id"), error="hypothesis_question_identity_invalid")
    hypothesis = _string(value.get("hypothesis"), error="hypothesis_text_invalid")
    digest = value.get("hypothesis_sha256")
    if digest is not None and _text_digest(hypothesis) != _digest_string(digest, error="hypothesis_digest_invalid"):
      raise QAError("hypothesis_digest_mismatch")
    if question_id in records:
      raise QAError("hypothesis_question_identity_duplicated")
    records[question_id] = {**value, "question_id": question_id, "hypothesis": hypothesis}
  return records, _file_digest(path)


def _references(args: argparse.Namespace) -> tuple[tuple[Reference, ...], dict[str, Any]]:
  if args.references is not None:
    if not args.sha256 or not args.revision:
      raise QAError("custom_reference_identity_required")
    path = args.references
    sha256 = _digest_string(args.sha256, error="custom_reference_digest_invalid")
    revision = _string(args.revision, error="custom_reference_revision_invalid")
    repository = "custom"
    size = path.stat().st_size if path.is_file() else -1
  else:
    name = args.dataset or "s"
    spec = longmem._DATASETS[name]
    path = args.data / spec["file"]
    sha256 = spec["sha256"]
    revision = longmem._DATASET_REVISION
    repository = longmem._DATASET_REPOSITORY
    size = spec["size"]
  try:
    longmem._verify_file(path, sha256=sha256, size=size)
  except longmem.LongMemError as exc:
    raise QAError(str(exc)) from exc
  try:
    with path.open(encoding="utf-8") as source:
      values = json.load(source)
  except (OSError, json.JSONDecodeError) as exc:
    raise QAError("reference_file_invalid") from exc
  if not isinstance(values, list):
    raise QAError("reference_file_invalid")
  references = tuple(_reference(value) for value in values)
  _unique((reference.question_id for reference in references), error="reference_question_identity_duplicated")
  return references, {
    "repository": repository,
    "revision": revision,
    "file": path.name,
    "sha256": sha256,
    "size": size,
  }


def _reference(value: Any) -> Reference:
  if not isinstance(value, dict):
    raise QAError("reference_record_invalid")
  question_id = _string(value.get("question_id"), error="reference_question_identity_invalid")
  question_type = _string(value.get("question_type"), error="reference_question_type_invalid")
  if question_type not in _TASKS:
    raise QAError("reference_question_type_invalid")
  answer = value.get("answer")
  if not isinstance(answer, str | int) or isinstance(answer, bool):
    raise QAError("reference_answer_invalid")
  answer = str(answer)
  if not answer.strip():
    raise QAError("reference_answer_invalid")
  return Reference(
    question_id=question_id,
    question_type=question_type,
    question=_string(value.get("question"), error="reference_question_invalid"),
    answer=answer,
    abstention="_abs" in question_id,
  )


def _existing(path: Path, *, kind: str) -> dict[str, dict[str, Any]]:
  if not path.exists():
    return {}
  values = _jsonl(path, error=f"{kind}_resume_invalid")
  records: dict[str, dict[str, Any]] = {}
  for value in values:
    record = _resume_record(value, kind=kind)
    question_id = record["question_id"]
    if question_id in records:
      raise QAError(f"{kind}_resume_invalid")
    records[question_id] = record
  return records


def _validate_existing(records: dict[str, dict[str, Any]], expected: dict[str, str], *, error: str) -> None:
  if any(question_id not in expected or record.get("request_id") != expected[question_id] for question_id, record in records.items()):
    raise QAError(error)


def _resume_record(value: Any, *, kind: str) -> dict[str, Any]:
  error = f"{kind}_resume_invalid"
  if not isinstance(value, dict):
    raise QAError(error)
  schema = {
    "hypothesis": "agos-memory-lab-hypothesis-v1",
    "evaluation": "agos-memory-lab-evaluation-v1",
  }.get(kind)
  if schema is None or value.get("schema") != schema:
    raise QAError(error)
  question_id = _resume_string(value.get("question_id"), error=error)
  _resume_digest(value.get("request_id"), error=error)
  _resume_usage(value.get("usage"), error=error)
  _resume_cost(value.get("cost"), error=error)
  _resume_duration(value.get("latency_seconds"), error=error)
  if kind == "hypothesis":
    _resume_digest(value.get("context_run_id"), error=error)
    _resume_digest(value.get("context_sha256"), error=error)
    hypothesis = _resume_string(value.get("hypothesis"), error=error)
    if _text_digest(hypothesis) != _resume_digest(value.get("hypothesis_sha256"), error=error):
      raise QAError(error)
    _resume_actor(value.get("reader"), error=error)
  else:
    if value.get("question_type") not in _TASKS or not isinstance(value.get("abstention"), bool):
      raise QAError(error)
    _resume_digest(value.get("hypothesis_sha256"), error=error)
    strict = value.get("strict_label")
    if not isinstance(value.get("official_label"), bool) or (strict is not None and not isinstance(strict, bool)):
      raise QAError(error)
    response = _resume_string(value.get("judge_response"), error=error)
    if _labels(response) != (value["official_label"], value["strict_label"]):
      raise QAError(error)
    _resume_actor(value.get("judge"), error=error)
  return {**value, "question_id": question_id}


def _resume_actor(value: Any, *, error: str) -> None:
  if not isinstance(value, dict):
    raise QAError(error)
  _resume_string(value.get("requested_model"), error=error)
  _resume_string(value.get("response_model"), error=error)


def _resume_usage(value: Any, *, error: str) -> None:
  try:
    bounded.validate_usage(value, error=error)
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc


def _resume_cost(value: Any, *, error: str) -> None:
  try:
    bounded.validate_cost(value, error=error)
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc


def _resume_duration(value: Any, *, error: str) -> None:
  try:
    bounded.validate_duration(value, error=error)
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc


def _resume_string(value: Any, *, error: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise QAError(error)
  return value


def _resume_digest(value: Any, *, error: str) -> str:
  if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
    raise QAError(error)
  return value


def _validate_record_cost(
  record: dict[str, Any], *, prompt: str, config: ChatConfig, error: str
) -> None:
  try:
    bounded.validate(
      record,
      prompt=prompt,
      config=config,
      overhead=_CHAT_ENVELOPE_TOKENS,
      error=error,
    )
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc


def _reader_id(context: Context, config: ChatConfig) -> str:
  return _digest(
    {
      "prompt": _PROMPT_REVISION,
      "context_run_id": context.run_id,
      "context_sha256": context.context_sha256,
      "question": context.question,
      "question_date": context.question_date,
      "chat": _request_value(config),
    }
  )


def _judge_id(reference: Reference, hypothesis: str, config: ChatConfig) -> str:
  return _digest(
    {
      "prompt": _JUDGE_REVISION,
      "question_id": reference.question_id,
      "question_type": reference.question_type,
      "question": reference.question,
      "answer": reference.answer,
      "abstention": reference.abstention,
      "hypothesis_sha256": _text_digest(hypothesis),
      "chat": _request_value(config),
    }
  )


def _run_receipt(
  *,
  kind: str,
  source: dict[str, Any],
  config: ChatConfig,
  records: list[dict[str, Any]],
  output: Path,
  scores: dict[str, Any] | None = None,
) -> dict[str, Any]:
  semantic = {
    "schema": f"agos-memory-lab-{kind}-v4",
    "benchmark_revision": longmem._BENCHMARK_REVISION,
    "source": source,
    "config": {
      "request": _request_value(config),
      "execution": _execution_value(config),
    },
    "prompt_revision": _PROMPT_REVISION if kind == "read" else _JUDGE_REVISION,
    "cases": [
      {
        "question_id": record["question_id"],
        "request_id": record["request_id"],
        "result_sha256": _record_result_digest(record),
        "response_model": record["reader" if kind == "read" else "judge"]["response_model"],
      }
      for record in records
    ],
  }
  if scores is not None:
    semantic["scores"] = scores
  run_id = _digest(semantic)
  receipt = {
    **semantic,
    "run_id": run_id,
    "summary": {**_usage_summary(records, config=config), **(scores or {})},
    "output": {"file": output.name, "sha256": _file_digest(output)},
  }
  return {**receipt, "receipt_sha256": _digest(receipt)}


def _usage_summary(records: list[dict[str, Any]], *, config: ChatConfig) -> dict[str, Any]:
  input_tokens = sum(record["usage"]["input_tokens"] or 0 for record in records)
  output_tokens = sum(record["usage"]["output_tokens"] or 0 for record in records)
  missing = sum(record["usage"]["total_tokens"] is None for record in records)
  estimated = [record["cost"]["estimated_usd"] for record in records]
  return {
    "cases": len(records),
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "usage_missing": missing,
    "estimated_cost_usd": round(sum(estimated), 8) if all(value is not None for value in estimated) else None,
    "reserved_cost_usd": round(sum(record["cost"]["reserved_usd"] for record in records), 8),
    "latency_seconds": round(sum(record["latency_seconds"] for record in records), 6),
  }


def _scores(records: list[dict[str, Any]]) -> dict[str, Any]:
  by_type = {
    task: [record for record in records if record["question_type"] == task]
    for task in _TASKS
  }
  populated = {task: values for task, values in by_type.items() if values}
  return {
    "accuracy": _accuracy(records),
    "task_accuracy": round(fmean(_accuracy(values) for values in populated.values()), 8) if populated else 0.0,
    "abstention_accuracy": _accuracy([record for record in records if record["abstention"]]),
    "strict_parse_failures": sum(record["strict_label"] is None for record in records),
    "strict_parse_disagreements": sum(
      record["strict_label"] is not None and record["strict_label"] != record["official_label"]
      for record in records
    ),
    "by_type": {
      task: {"cases": len(values), "accuracy": _accuracy(values)}
      for task, values in populated.items()
    },
  }


def _accuracy(records: list[dict[str, Any]]) -> float:
  return round(fmean(record["official_label"] for record in records), 8) if records else 0.0


def _labels(value: str) -> tuple[bool, bool | None]:
  normalized = re.findall(r"[a-z]+", value.lower())
  strict = normalized[0] == "yes" if normalized and normalized[0] in {"yes", "no"} else None
  return "yes" in value.lower(), strict


def _usage(result: ChatResult) -> dict[str, int | None]:
  return bounded.usage(result)


def _check_cost(records: Any, *, prompt: str, config: ChatConfig) -> None:
  try:
    bounded.check(
      records,
      prompt=prompt,
      config=config,
      overhead=_CHAT_ENVELOPE_TOKENS,
    )
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc


def _cost(result: ChatResult, *, prompt: str, config: ChatConfig) -> dict[str, float | None]:
  try:
    return bounded.cost(
      result,
      prompt=prompt,
      config=config,
      overhead=_CHAT_ENVELOPE_TOKENS,
    )
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc


def _estimated_cost(result: ChatResult, *, config: ChatConfig) -> float | None:
  return bounded.estimate(result, config=config)


def _reserved_cost(prompt: str, *, config: ChatConfig) -> float:
  try:
    return bounded.reserve(prompt, config=config, overhead=_CHAT_ENVELOPE_TOKENS)
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc


def _request_value(config: ChatConfig) -> dict[str, Any]:
  return bounded.request(config)


def _execution_value(config: ChatConfig) -> dict[str, Any]:
  return bounded.execution(config)


def _record_result_digest(record: dict[str, Any]) -> str:
  if "hypothesis_sha256" in record and "official_label" not in record:
    return record["hypothesis_sha256"]
  return _digest(
    {
      "hypothesis_sha256": record.get("hypothesis_sha256"),
      "official_label": record.get("official_label"),
      "strict_label": record.get("strict_label"),
      "judge_response": record.get("judge_response"),
    }
  )


def _window(values: tuple[Context, ...], *, offset: int, limit: int) -> tuple[Context, ...]:
  if offset < 0 or limit < 0:
    raise QAError("context_window_invalid")
  selected = values[offset : None if limit == 0 else offset + limit]
  if not selected:
    raise QAError("context_window_empty")
  return selected


def _mapping_window(
  values: dict[str, dict[str, Any]], *, offset: int, limit: int
) -> dict[str, dict[str, Any]]:
  if offset < 0 or limit < 0:
    raise QAError("hypothesis_window_invalid")
  items = tuple(values.items())[offset : None if limit == 0 else offset + limit]
  if not items:
    raise QAError("hypothesis_window_empty")
  return dict(items)


def _jsonl(path: Path, *, error: str) -> list[Any]:
  if not path.is_file():
    raise QAError(error)
  values = []
  try:
    with path.open(encoding="utf-8") as source:
      for line in source:
        if line.strip():
          values.append(json.loads(line))
  except (OSError, json.JSONDecodeError) as exc:
    raise QAError(error) from exc
  if not values:
    raise QAError(error)
  return values


def _distinct_paths(*paths: Path) -> None:
  resolved = [path.resolve() for path in paths]
  if len(resolved) != len(set(resolved)):
    raise QAError("input_output_path_conflict")


def _recover_pending(path: Path, *, records: dict[str, dict[str, Any]]) -> None:
  try:
    value = bounded.pending(path, error="chat_request_outcome_unknown")
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc
  if value is None:
    return
  question_id = value.get("question_id")
  request_id = value.get("request_id")
  record = records.get(question_id) if isinstance(question_id, str) else None
  if (
    value.get("schema") != "agos-memory-lab-pending-v1"
    or not isinstance(request_id, str)
    or record is None
    or record["request_id"] != request_id
  ):
    raise QAError("chat_request_outcome_unknown")
  _clear_pending(path)


def _begin_pending(path: Path, *, question_id: str, request_id: str, prompt: str) -> None:
  value = {
    "schema": "agos-memory-lab-pending-v1",
    "question_id": question_id,
    "request_id": request_id,
    "prompt_sha256": _text_digest(prompt),
  }
  try:
    bounded.begin(
      path,
      value,
      unknown="chat_request_outcome_unknown",
      failed="chat_pending_write_failed",
    )
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc


def _clear_pending(path: Path) -> None:
  try:
    bounded.clear(path, error="chat_pending_clear_failed")
  except bounded.CallError as exc:
    raise QAError(str(exc)) from exc


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix(f"{path.suffix}.part")
  partial.write_bytes(
    "".join(f"{json.dumps(value, sort_keys=True)}\n" for value in values).encode("utf-8")
  )
  os.replace(partial, path)


def _write(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix(f"{path.suffix}.part")
  partial.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
  os.replace(partial, path)


def _receipt_path(path: Path) -> Path:
  return path.with_suffix(f"{path.suffix}.receipt.json")


def _pending_path(path: Path) -> Path:
  return path.with_suffix(f"{path.suffix}.pending.json")


def _string(value: Any, *, error: str, empty: bool = False) -> str:
  if not isinstance(value, str) or (not empty and not value.strip()):
    raise QAError(error)
  return value


def _digest_string(value: Any, *, error: str) -> str:
  if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
    raise QAError(error)
  return value


def _unique(values: Any, *, error: str) -> None:
  items = tuple(values)
  if len(items) != len(set(items)):
    raise QAError(error)


def _one(values: Any) -> str:
  items = set(values)
  if len(items) != 1:
    raise QAError("source_run_identity_mixed")
  return next(iter(items))


def _text_digest(value: str) -> str:
  return hashlib.sha256(value.encode()).hexdigest()


def _file_digest(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _digest(value: Any) -> str:
  return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
  main()
