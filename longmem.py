# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "agos-memory==0.1.0",
#   "fastembed==0.7.4",
#   "qdrant-client[fastembed]==1.17.1",
#   "rank-bm25==0.2.2",
# ]
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from agos_memory.select import select
from agos_memory.support import source_digest
from agos_memory.types import (
  Omitted,
  Selected,
  SelectionItem,
  SelectionLimits,
  SelectionPolicy,
  SelectionPriority,
  SelectionRoute,
)
from rank_bm25 import BM25Okapi


_DATASET_REPOSITORY = "xiaowu0162/longmemeval-cleaned"
_DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
_BENCHMARK_REPOSITORY = "xiaowu0162/LongMemEval"
_BENCHMARK_REVISION = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
_DATASETS = {
  "s": {
    "file": "longmemeval_s_cleaned.json",
    "sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    "size": 277_383_467,
  },
  "oracle": {
    "file": "longmemeval_oracle.json",
    "sha256": "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
    "size": 15_388_478,
  },
}
_CUTS = (1, 3, 5, 10, 50)
_Retriever = Literal["none", "recent", "full", "lexical", "oracle", "qdrant-dense", "qdrant-hybrid"]


@dataclass(frozen=True, slots=True)
class Session:
  source_id: str
  benchmark_id: str
  date: str
  at: datetime
  text: str
  content: str


@dataclass(frozen=True, slots=True)
class Case:
  question_id: str
  question_type: str
  question: str
  question_date: str
  asked_at: datetime
  sessions: tuple[Session, ...]
  answer_session_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Hit:
  source_id: str


class LongMemError(Exception):
  pass


def main() -> None:
  parser = _parser()
  args = parser.parse_args()
  try:
    if args.command == "fetch":
      _fetch(args)
    elif args.command == "run":
      _run(args)
    else:
      parser.error("command_required")
  except LongMemError as exc:
    raise SystemExit(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run reproducible LongMemEval retrieval experiments.")
  commands = parser.add_subparsers(dest="command")

  fetch = commands.add_parser("fetch", help="Download and verify one pinned official dataset.")
  fetch.add_argument("--dataset", choices=tuple(_DATASETS), default="s")
  fetch.add_argument("--data", type=Path, default=Path("data"))

  run = commands.add_parser("run", help="Run retrieval and governed selection.")
  source = run.add_mutually_exclusive_group()
  source.add_argument("--dataset", choices=tuple(_DATASETS), help="Pinned corpus; defaults to s.")
  source.add_argument("--file", type=Path, help="Custom LongMemEval-shaped JSON file.")
  run.add_argument("--sha256", help="Required SHA-256 for --file.")
  run.add_argument("--revision", help="Required immutable identity for --file.")
  run.add_argument("--data", type=Path, default=Path("data"))
  run.add_argument("--retriever", choices=_Retriever.__args__, default="lexical")
  run.add_argument("--offset", type=int, default=0, help="First corpus case; defaults to 0.")
  run.add_argument("--limit", type=int, default=0, help="Number of cases; 0 means all.")
  run.add_argument("--candidates", type=int, default=50, help="Raw candidates per case; maximum 100.")
  run.add_argument("--top-k", type=int, default=10, help="Maximum kernel-selected sessions.")
  run.add_argument("--chars", type=int, default=180_000, help="Maximum kernel context characters.")
  run.add_argument("--lexical-weight", type=int, default=0, help="Kernel lexical reranking weight.")
  run.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="FastEmbed dense model.")
  run.add_argument("--cache", type=Path, default=Path("data/models"))
  run.add_argument("--out", type=Path)
  run.add_argument("--contexts", type=Path, help="Optional gold-free selected context JSONL.")
  return parser


def _fetch(args: argparse.Namespace) -> None:
  spec = _DATASETS[args.dataset]
  target = args.data / spec["file"]
  if target.exists():
    _verify_file(target, sha256=spec["sha256"], size=spec["size"])
    print(json.dumps({"dataset": args.dataset, "path": str(target), "status": "current"}, sort_keys=True))
    return

  args.data.mkdir(parents=True, exist_ok=True)
  partial = target.with_suffix(f"{target.suffix}.part")
  url = (
    f"https://huggingface.co/datasets/{_DATASET_REPOSITORY}/resolve/"
    f"{_DATASET_REVISION}/{spec['file']}?download=true"
  )
  request = urllib.request.Request(url, headers={"User-Agent": "agos-memory-lab/0.0.0"})
  digest = hashlib.sha256()
  size = 0
  try:
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
      while chunk := response.read(1024 * 1024):
        output.write(chunk)
        digest.update(chunk)
        size += len(chunk)
  except OSError as exc:
    raise LongMemError(f"dataset_download_failed:{exc}") from exc
  if size != spec["size"] or digest.hexdigest() != spec["sha256"]:
    partial.unlink(missing_ok=True)
    raise LongMemError("dataset_download_identity_mismatch")
  os.replace(partial, target)
  print(json.dumps({"dataset": args.dataset, "path": str(target), "status": "downloaded"}, sort_keys=True))


def _run(args: argparse.Namespace) -> None:
  _validate_run(args)
  started = time.perf_counter()
  loaded = time.perf_counter()
  source = _source(args)
  _verify_file(source["path"], sha256=source["sha256"], size=source["size"])
  cases = _load(source["path"])
  cases = cases[args.offset : None if args.limit == 0 else args.offset + args.limit]
  if not cases:
    raise LongMemError("dataset_selection_empty")
  load_seconds = time.perf_counter() - loaded

  qdrant = None
  index_seconds = 0.0
  if args.retriever.startswith("qdrant-"):
    indexed = time.perf_counter()
    qdrant = _Qdrant(
      cases,
      hybrid=args.retriever == "qdrant-hybrid",
      model=args.model,
      cache=args.cache,
    )
    index_seconds = time.perf_counter() - indexed

  query_seconds = 0.0
  kernel_seconds = 0.0
  retriever_identity = qdrant.identity if qdrant is not None else _retriever_identity(args.retriever)
  results: list[dict[str, Any]] = []
  contexts: list[dict[str, Any]] = []
  try:
    for case in cases:
      queried = time.perf_counter()
      hits = (
        qdrant.retrieve(case, limit=min(args.candidates, len(case.sessions)))
        if qdrant is not None
        else _retrieve(case, retriever=args.retriever, limit=args.candidates)
      )
      query_seconds += time.perf_counter() - queried

      governed = time.perf_counter()
      kernel = _govern(
        case,
        hits,
        retriever=args.retriever,
        top_k=args.top_k,
        chars=args.chars,
        lexical_weight=args.lexical_weight,
      )
      kernel_seconds += time.perf_counter() - governed
      raw_ids = tuple(hit.source_id for hit in hits)
      raw_benchmark_ids = tuple(_benchmark_id(case, source_id) for source_id in raw_ids)
      selected_benchmark_ids = tuple(
        _benchmark_id(case, source_id) for source_id in kernel["selected_occurrence_ids"]
      )
      contexts.append(
        {
          "question_id": case.question_id,
          "question": case.question,
          "question_date": case.question_date,
          "selected_occurrence_ids": kernel["selected_occurrence_ids"],
          "context": kernel["content"],
          "context_sha256": kernel["receipt"]["content_sha256"],
        }
      )
      results.append(
        {
          "question_id": case.question_id,
          "question_type": case.question_type,
          "abstention": case.question_id.endswith("_abs"),
          "answer_session_ids": case.answer_session_ids,
          "retrieved_occurrence_ids": raw_ids,
          "retrieved_session_ids": raw_benchmark_ids,
          "selected_occurrence_ids": kernel["selected_occurrence_ids"],
          "selected_session_ids": selected_benchmark_ids,
          "metrics": {
            "retrieval": _metrics(raw_benchmark_ids, case.answer_session_ids),
            "kernel": _metrics(selected_benchmark_ids, case.answer_session_ids),
          },
          "kernel": kernel["receipt"],
        }
      )
  finally:
    if qdrant is not None:
      qdrant.close()

  semantic = {
    "schema": "agos-memory-lab-longmem-v1",
    "benchmark": {
      "repository": _BENCHMARK_REPOSITORY,
      "revision": _BENCHMARK_REVISION,
    },
    "dataset": {
      "repository": source["repository"],
      "revision": source["revision"],
      "file": source["path"].name,
      "sha256": source["sha256"],
      "size": source["size"],
    },
    "kernel": version("agos-memory"),
    "config": {
      "retriever": args.retriever,
      "offset": args.offset,
      "limit": len(cases),
      "candidates": args.candidates,
      "top_k": args.top_k,
      "chars": args.chars,
      "lexical_weight": args.lexical_weight,
      "retriever_identity": retriever_identity,
    },
    "summary": _summary(results),
    "cases": results,
  }
  run_id = _digest(semantic)
  receipt = {
    **semantic,
    "run_id": run_id,
    "measurements": {
      "load_seconds": round(load_seconds, 6),
      "index_seconds": round(index_seconds, 6),
      "query_seconds": round(query_seconds, 6),
      "kernel_seconds": round(kernel_seconds, 6),
      "total_seconds": round(time.perf_counter() - started, 6),
    },
  }
  receipt = {**receipt, "receipt_sha256": _digest(receipt)}
  out = args.out or Path("runs") / f"{args.retriever}-{run_id[:12]}.json"
  _write(out, receipt)
  if args.contexts is not None:
    _write_jsonl(args.contexts, contexts)
  print(
    json.dumps(
      {
        "out": str(out),
        "contexts": str(args.contexts) if args.contexts is not None else None,
        "run_id": run_id,
        "summary": receipt["summary"],
      },
      indent=2,
      sort_keys=True,
    )
  )


def _source(args: argparse.Namespace) -> dict[str, Any]:
  if args.file is not None:
    if not args.sha256 or not args.revision:
      raise LongMemError("custom_dataset_identity_required")
    return {
      "repository": "custom",
      "revision": args.revision,
      "path": args.file,
      "sha256": args.sha256,
      "size": args.file.stat().st_size if args.file.exists() else -1,
    }
  name = args.dataset or "s"
  spec = _DATASETS[name]
  return {
    "repository": _DATASET_REPOSITORY,
    "revision": _DATASET_REVISION,
    "path": args.data / spec["file"],
    "sha256": spec["sha256"],
    "size": spec["size"],
  }


def _retrieve(case: Case, *, retriever: _Retriever, limit: int) -> tuple[Hit, ...]:
  sessions = case.sessions
  if retriever == "none":
    return ()
  if retriever == "recent":
    ordered = sorted(enumerate(sessions), key=lambda item: (-item[1].at.timestamp(), item[0]))
  elif retriever == "full":
    ordered = list(enumerate(sessions))
    limit = len(sessions)
    if limit > 100:
      raise LongMemError("full_retriever_route_limit_exceeded")
  elif retriever == "lexical":
    bm25 = BM25Okapi([session.text.split(" ") for session in sessions])
    scores = bm25.get_scores(case.question.split(" "))
    ordered = [(index, sessions[index]) for index in scores.argsort()[::-1]]
  elif retriever == "oracle":
    answer = set(case.answer_session_ids)
    ordered = sorted(enumerate(sessions), key=lambda item: (item[1].benchmark_id not in answer, item[0]))
  else:
    raise LongMemError("qdrant_dependency_missing:run_with_uv_script")
  return tuple(Hit(session.source_id) for _, session in ordered[:limit])


def _rrf(case: Case, *rankings: tuple[Hit, ...], limit: int) -> tuple[Hit, ...]:
  scores: dict[str, float] = {}
  for ranking in rankings:
    for rank, hit in enumerate(ranking, start=1):
      scores[hit.source_id] = scores.get(hit.source_id, 0.0) + 1 / (60 + rank)
  order = {session.source_id: index for index, session in enumerate(case.sessions)}
  source_ids = sorted(scores, key=lambda source_id: (-scores[source_id], order[source_id]))
  return tuple(Hit(source_id) for source_id in source_ids[:limit])


def _govern(
  case: Case,
  hits: tuple[Hit, ...],
  *,
  retriever: _Retriever,
  top_k: int,
  chars: int,
  lexical_weight: int,
) -> dict[str, Any]:
  by_id = {session.source_id: session for session in case.sessions}
  if any(hit.source_id not in by_id for hit in hits):
    raise LongMemError("retrieval_result_scope_mismatch")
  items = tuple(
    SelectionItem(
      source="session",
      source_id=hit.source_id,
      partition="history",
      kind="session",
      text=by_id[hit.source_id].text,
      content=(
        f"Session Date: {by_id[hit.source_id].date}\nSession Content:\n{by_id[hit.source_id].content}"
        if by_id[hit.source_id].content
        else ""
      ),
      updated_at=by_id[hit.source_id].at,
      available_at=None,
      revision=by_id[hit.source_id].date,
      source_digest=source_digest(by_id[hit.source_id].content),
    )
    for hit in hits
  )
  routes = tuple(
    SelectionRoute(
      source="session",
      source_id=hit.source_id,
      lane=retriever,
      rank=rank,
      signal=retriever,
    )
    for rank, hit in enumerate(hits, start=1)
  )
  selection = select(
    items,
    routes=routes,
    query=case.question,
    limits=SelectionLimits(max_items=top_k, max_chars=chars),
    policy=SelectionPolicy(
      partitions=(SelectionPriority(label="history", score=0),),
      kinds=(SelectionPriority(label="session", score=0),),
      source_order=("session",),
      route_order=(retriever,),
      lexical_weight=lexical_weight,
      route_rank_ceiling=100,
    ),
    now=case.asked_at,
    include_paths=True,
  )
  outcomes = tuple(_outcome(outcome) for outcome in selection.outcomes)
  selected = tuple(
    outcome.candidate.source_id
    for outcome in sorted(selection.selected, key=lambda item: item.rank)
  )
  return {
    "selected_occurrence_ids": selected,
    "content": selection.content,
    "receipt": {
      "content_chars": len(selection.content),
      "content_sha256": hashlib.sha256(selection.content.encode()).hexdigest(),
      "source_count": selection.source_count,
      "included_count": selection.included_count,
      "truncated": selection.truncated,
      "outcomes": outcomes,
    },
  }


def _outcome(outcome: Selected | Omitted) -> dict[str, Any]:
  candidate = outcome.candidate
  value = {
    "source_id": candidate.source_id,
    "candidate_rank": candidate.candidate_rank,
    "score": candidate.score,
    "text_hash": candidate.text_hash,
    "paths": tuple(
      {
        "lane": path.lane,
        "rank": path.rank,
        "signal": path.signal,
        "relation": path.relation,
      }
      for path in candidate.paths
    ),
  }
  if isinstance(outcome, Selected):
    return {**value, "selected": True, "rank": outcome.rank, "content_chars": outcome.content_chars}
  return {**value, "selected": False, "reason": outcome.reason}


def _metrics(ranked: tuple[str, ...], correct: tuple[str, ...]) -> dict[str, float] | None:
  if not correct:
    return None
  gold = set(correct)
  metrics: dict[str, float] = {}
  for cut in _CUTS:
    recalled = set(ranked[:cut])
    relevances = tuple(1 if source_id in gold else 0 for source_id in ranked[:cut])
    ideal = (1,) * min(len(gold), cut)
    metrics[f"recall_any@{cut}"] = float(bool(recalled & gold))
    metrics[f"recall_all@{cut}"] = float(gold <= recalled)
    metrics[f"ndcg_any@{cut}"] = round(_dcg(relevances) / _dcg(ideal), 8) if ideal else 0.0
  return metrics


def _dcg(relevances: tuple[int, ...]) -> float:
  if not relevances:
    return 0.0
  return float(relevances[0]) + sum(value / math.log2(index + 1) for index, value in enumerate(relevances[1:], start=1))


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
  eligible = [result for result in results if not result["abstention"] and result["answer_session_ids"]]
  by_type: dict[str, list[dict[str, Any]]] = {}
  for result in eligible:
    by_type.setdefault(result["question_type"], []).append(result)
  return {
    "cases": len(results),
    "eligible": len(eligible),
    "ignored_abstention": sum(result["abstention"] for result in results),
    "ignored_missing_target": sum(not result["abstention"] and not result["answer_session_ids"] for result in results),
    "retrieval": _average(eligible, stage="retrieval"),
    "kernel": _average(eligible, stage="kernel"),
    "selection": _selection_summary(results),
    "by_type": {
      question_type: {
        "cases": len(group),
        "retrieval": _average(group, stage="retrieval"),
        "kernel": _average(group, stage="kernel"),
      }
      for question_type, group in sorted(by_type.items())
    },
  }


def _average(results: list[dict[str, Any]], *, stage: str) -> dict[str, float]:
  if not results:
    return {}
  names = results[0]["metrics"][stage]
  if names is None:
    return {}
  return {
    name: round(fmean(result["metrics"][stage][name] for result in results), 8)
    for name in names
  }


def _selection_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
  reasons: dict[str, int] = {}
  for result in results:
    for outcome in result["kernel"]["outcomes"]:
      reason = "selected" if outcome["selected"] else outcome["reason"]
      reasons[reason] = reasons.get(reason, 0) + 1
  return {
    "truncated_cases": sum(result["kernel"]["truncated"] for result in results),
    "mean_candidates": round(fmean(result["kernel"]["source_count"] for result in results), 8),
    "mean_selected": round(fmean(result["kernel"]["included_count"] for result in results), 8),
    "mean_content_chars": round(fmean(result["kernel"]["content_chars"] for result in results), 8),
    "outcomes": dict(sorted(reasons.items())),
  }


class _Qdrant:
  def __init__(
    self,
    cases: tuple[Case, ...],
    *,
    hybrid: bool,
    model: str,
    cache: Path,
  ) -> None:
    try:
      from fastembed import TextEmbedding
      from qdrant_client import QdrantClient, models
    except ImportError as exc:
      raise LongMemError("qdrant_dependency_missing:run_with_uv_script") from exc

    cache.mkdir(parents=True, exist_ok=True)
    self._models = models
    self._client = QdrantClient(":memory:")
    self._dense = TextEmbedding(model, cache_dir=str(cache))
    self._hybrid = hybrid
    self._collection = "longmem"
    self.identity = {
      "qdrant_client": version("qdrant-client"),
      "fastembed": version("fastembed"),
      "runtime": {
        "numpy": version("numpy"),
        "onnxruntime": version("onnxruntime"),
        "tokenizers": version("tokenizers"),
      },
      "dense": _model_identity(self._dense),
      "fusion": (
        {"algorithm": "rrf", "k": 60, "lexical": _retriever_identity("lexical")}
        if hybrid
        else None
      ),
    }

    rows = tuple((case.question_id, session) for case in cases for session in case.sessions)
    self._client.create_collection(
      self._collection,
      vectors_config={
        "dense": models.VectorParams(size=self._dense.embedding_size, distance=models.Distance.COSINE)
      },
    )
    for start in range(0, len(rows), 256):
      batch = rows[start : start + 256]
      dense_vectors = self._dense.embed(session.text for _, session in batch)
      points = [
        models.PointStruct(
          id=start + offset,
          payload={"question_id": question_id, "source_id": session.source_id},
          vector={"dense": vector.tolist()},
        )
        for offset, ((question_id, session), vector) in enumerate(
          zip(batch, dense_vectors, strict=True)
        )
      ]
      self._client.upsert(self._collection, points=points, wait=True)

  def retrieve(self, case: Case, *, limit: int) -> tuple[Hit, ...]:
    models = self._models
    dense = next(iter(self._dense.query_embed(case.question))).tolist()
    query_filter = models.Filter(
      must=[models.FieldCondition(key="question_id", match=models.MatchValue(value=case.question_id))]
    )
    response = self._client.query_points(
      self._collection,
      query=dense,
      using="dense",
      query_filter=query_filter,
      limit=limit,
      with_payload=["question_id", "source_id"],
    )
    allowed = {session.source_id for session in case.sessions}
    payloads = tuple(point.payload or {} for point in response.points)
    if any(
      payload.get("question_id") != case.question_id or payload.get("source_id") not in allowed
      for payload in payloads
    ):
      raise LongMemError("qdrant_result_scope_mismatch")
    hits = tuple(Hit(str(payload["source_id"])) for payload in payloads)
    if len(hits) != len(set(hit.source_id for hit in hits)):
      raise LongMemError("qdrant_result_identity_duplicated")
    if not self._hybrid:
      return hits
    lexical = _retrieve(case, retriever="lexical", limit=limit)
    return _rrf(case, hits, lexical, limit=limit)

  def close(self) -> None:
    self._client.close()


def _load(path: Path) -> tuple[Case, ...]:
  try:
    with path.open() as source:
      values = json.load(source)
  except (OSError, json.JSONDecodeError) as exc:
    raise LongMemError(f"dataset_read_failed:{exc}") from exc
  if not isinstance(values, list):
    raise LongMemError("dataset_root_invalid")
  cases = tuple(_case(value) for value in values)
  ids = tuple(case.question_id for case in cases)
  if len(ids) != len(set(ids)):
    raise LongMemError("dataset_question_identity_duplicated")
  return cases


def _case(value: Any) -> Case:
  if not isinstance(value, dict):
    raise LongMemError("dataset_case_invalid")
  question_id = _text(value.get("question_id"), "dataset_question_id_invalid")
  question_type = _text(value.get("question_type"), "dataset_question_type_invalid")
  question = _text(value.get("question"), "dataset_question_invalid")
  question_date = _text(value.get("question_date"), "dataset_question_date_invalid")
  session_ids = _texts(value.get("haystack_session_ids"), "dataset_session_ids_invalid")
  dates = _texts(value.get("haystack_dates"), "dataset_session_dates_invalid")
  sessions = value.get("haystack_sessions")
  if not isinstance(sessions, list) or len(session_ids) != len(dates) or len(session_ids) != len(sessions):
    raise LongMemError("dataset_session_shape_invalid")
  rows = tuple(
    _session(index=index, benchmark_id=source_id, date=date, turns=turns)
    for index, (source_id, date, turns) in enumerate(zip(session_ids, dates, sessions, strict=True))
  )
  answers = _texts(value.get("answer_session_ids"), "dataset_answer_session_ids_invalid")
  if not set(answers) <= set(session_ids):
    raise LongMemError("dataset_answer_session_missing")
  return Case(
    question_id=question_id,
    question_type=question_type,
    question=question,
    question_date=question_date,
    asked_at=_date(question_date),
    sessions=rows,
    answer_session_ids=answers,
  )


def _session(*, index: int, benchmark_id: str, date: str, turns: Any) -> Session:
  text, content = _session_text(turns)
  return Session(
    source_id=f"{index}:{benchmark_id}",
    benchmark_id=benchmark_id,
    date=date,
    at=_date(date),
    text=text,
    content=content,
  )


def _session_text(value: Any) -> tuple[str, str]:
  if not isinstance(value, list):
    raise LongMemError("dataset_session_invalid")
  user: list[str] = []
  dialogue: list[str] = []
  for turn in value:
    if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}:
      raise LongMemError("dataset_turn_invalid")
    content = turn.get("content")
    if not isinstance(content, str):
      raise LongMemError("dataset_turn_content_invalid")
    dialogue.append(f"{turn['role']}: {content}")
    if turn["role"] == "user":
      user.append(content)
  return " ".join(user), "\n".join(dialogue)


def _date(value: str) -> datetime:
  try:
    return datetime.strptime(f"{value[:10]} {value[-5:]}", "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
  except ValueError as exc:
    raise LongMemError("dataset_date_invalid") from exc


def _text(value: Any, error: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise LongMemError(error)
  return value


def _texts(value: Any, error: str) -> tuple[str, ...]:
  if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
    raise LongMemError(error)
  return tuple(value)


def _verify_file(path: Path, *, sha256: str, size: int) -> None:
  if not path.is_file():
    raise LongMemError(f"dataset_missing:{path}")
  if path.stat().st_size != size:
    raise LongMemError("dataset_size_mismatch")
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(1024 * 1024):
      digest.update(chunk)
  if digest.hexdigest() != sha256:
    raise LongMemError("dataset_sha256_mismatch")


def _validate_run(args: argparse.Namespace) -> None:
  for value, error in (
    (args.offset, "offset_invalid"),
    (args.limit, "limit_invalid"),
  ):
    if value < 0:
      raise LongMemError(error)
  for value, error in (
    (args.candidates, "candidate_limit_invalid"),
    (args.top_k, "top_k_invalid"),
    (args.chars, "char_limit_invalid"),
  ):
    if value < 1:
      raise LongMemError(error)
  if args.candidates > 100:
    raise LongMemError("candidate_limit_invalid")
  if args.top_k > 100:
    raise LongMemError("top_k_invalid")
  if args.lexical_weight < 0:
    raise LongMemError("lexical_weight_invalid")
  if args.top_k > args.candidates and args.retriever != "full":
    raise LongMemError("top_k_exceeds_candidates")


def _benchmark_id(case: Case, source_id: str) -> str:
  for session in case.sessions:
    if session.source_id == source_id:
      return session.benchmark_id
  raise LongMemError("selected_occurrence_missing")


def _write(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix(f"{path.suffix}.part")
  partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  os.replace(partial, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix(f"{path.suffix}.part")
  partial.write_text("".join(f"{json.dumps(value, sort_keys=True)}\n" for value in values))
  os.replace(partial, path)


def _digest(value: Any) -> str:
  return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _retriever_identity(retriever: _Retriever) -> dict[str, Any]:
  if retriever == "lexical":
    return {
      "package": "rank-bm25",
      "version": version("rank-bm25"),
      "numpy": version("numpy"),
      "algorithm": "BM25Okapi",
      "text": "user turns",
      "tie_order": "official descending argsort",
    }
  return {"algorithm": retriever}


def _model_identity(embedder: Any) -> dict[str, str]:
  value = getattr(embedder.model, "_model_dir", None)
  if not isinstance(value, (str, os.PathLike)):
    raise LongMemError("embedding_model_identity_missing")
  model_dir = Path(value)
  if not model_dir.is_dir():
    raise LongMemError("embedding_model_identity_missing")
  digest = hashlib.sha256()
  for path in sorted(path for path in model_dir.rglob("*") if path.is_file()):
    digest.update(str(path.relative_to(model_dir)).encode())
    digest.update(b"\0")
    with path.open("rb") as source:
      while chunk := source.read(1024 * 1024):
        digest.update(chunk)
  return {"name": embedder.model_name, "revision": model_dir.name, "sha256": digest.hexdigest()}


if __name__ == "__main__":
  main()
