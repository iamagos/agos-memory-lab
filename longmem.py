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
from typing import TYPE_CHECKING, Any, Literal

from agos_memory.retain import retain
from agos_memory.select import select
from agos_memory.support import source_digest, support
from agos_memory.types import (
  Omit,
  Omitted,
  ReopenedSource,
  RetentionPolicy,
  RetentionState,
  Selected,
  SelectionItem,
  SelectionLimits,
  SelectionPath,
  SelectionPolicy,
  SelectionPriority,
  SelectionRoute,
)
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
  import memory


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
_Source = Literal["sessions", "memories"]


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
  paths: tuple[SelectionPath, ...] = ()


@dataclass(frozen=True, slots=True)
class Ranking:
  lane: str
  hits: tuple[Hit, ...]


@dataclass(frozen=True, slots=True)
class Entry:
  source_id: str
  benchmark_id: str
  text: str
  at: datetime


@dataclass(frozen=True, slots=True)
class Memory:
  record: memory.Record
  session: Session


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
  run.add_argument("--source", choices=_Source.__args__, default="sessions")
  run.add_argument("--artifact", type=Path, help="Required source-linked artifact for memories.")
  run.add_argument("--artifact-sha256", help="Required artifact SHA-256 for memories.")
  run.add_argument("--retriever", choices=_Retriever.__args__, default="lexical")
  run.add_argument("--offset", type=int, default=0, help="First corpus case; defaults to 0.")
  run.add_argument("--limit", type=int, default=0, help="Number of cases; 0 means all.")
  run.add_argument("--candidates", type=int, default=50, help="Raw candidates per case; maximum 100.")
  run.add_argument("--top-k", type=int, default=10, help="Maximum kernel-selected sessions.")
  run.add_argument(
    "--episodes",
    type=int,
    default=0,
    help="Current raw-session candidates per memory case; defaults to 0.",
  )
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
  all_cases = _load(source["path"])
  artifact, memories_by_case = _load_memories(args, source=source, cases=all_cases)
  cases = all_cases[args.offset : None if args.limit == 0 else args.offset + args.limit]
  if not cases:
    raise LongMemError("dataset_selection_empty")
  entries_by_case = {
    case.question_id: (
      _session_entries(case)
      if artifact is None
      else _memory_entries(memories_by_case.get(case.question_id, ()))
    )
    for case in cases
  }
  episode_entries_by_case = (
    {case.question_id: _episode_entries(case) for case in cases}
    if args.episodes
    else {}
  )
  load_seconds = time.perf_counter() - loaded

  qdrant = None
  episode_qdrant = None
  index_seconds = 0.0
  if args.retriever.startswith("qdrant-"):
    indexed = time.perf_counter()
    qdrant = _Qdrant(
      cases,
      entries_by_case,
      hybrid=args.retriever == "qdrant-hybrid",
      model=args.model,
      cache=args.cache,
      text="user turns" if artifact is None else "memory text",
    )
    if args.episodes:
      try:
        episode_qdrant = _Qdrant(
          cases,
          episode_entries_by_case,
          hybrid=args.retriever == "qdrant-hybrid",
          model=args.model,
          cache=args.cache,
          text="user turns",
          embedder=qdrant.embedder,
        )
      except Exception:
        qdrant.close()
        raise
    index_seconds = time.perf_counter() - indexed

  query_seconds = 0.0
  kernel_seconds = 0.0
  primary_retriever_identity = (
    qdrant.identity
    if qdrant is not None
    else _retriever_identity(
      args.retriever,
      text="user turns" if artifact is None else "memory text",
    )
  )
  retriever_identity = (
    {
      "memories": primary_retriever_identity,
      "episodes": (
        episode_qdrant.identity
        if episode_qdrant is not None
        else _retriever_identity(args.retriever, text="user turns")
      ),
    }
    if args.episodes
    else primary_retriever_identity
  )
  results: list[dict[str, Any]] = []
  contexts: list[dict[str, Any]] = []
  try:
    for case in cases:
      entries = entries_by_case[case.question_id]
      queried = time.perf_counter()
      hits = (
        qdrant.retrieve(case, entries, limit=min(args.candidates, len(entries)))
        if qdrant is not None
        else _retrieve_entries(
          case,
          entries,
          retriever=args.retriever,
          limit=args.candidates,
        )
      )
      episode_entries = episode_entries_by_case.get(case.question_id, ())
      episode_hits = (
        (
          episode_qdrant.retrieve(
            case,
            episode_entries,
            limit=min(args.episodes, len(episode_entries)),
          )
          if episode_qdrant is not None
          else _retrieve_episodes(
            case,
            episode_entries,
            retriever=args.retriever,
            limit=args.episodes,
          )
        )
        if args.episodes
        else ()
      )
      query_seconds += time.perf_counter() - queried

      governed = time.perf_counter()
      if artifact is None:
        kernel = _govern(
          case,
          hits,
          retriever=args.retriever,
          top_k=args.top_k,
          chars=args.chars,
          lexical_weight=args.lexical_weight,
        )
      else:
        kernel = _govern_memory(
          case,
          memories_by_case.get(case.question_id, ()),
          hits,
          episode_hits=episode_hits,
          artifact=artifact,
          source=source,
          retriever=args.retriever,
          top_k=args.top_k,
          chars=args.chars,
          lexical_weight=args.lexical_weight,
        )
      kernel_seconds += time.perf_counter() - governed
      raw_ids = tuple(hit.source_id for hit in hits)
      if artifact is None:
        raw_benchmark_ids = tuple(_benchmark_id(case, source_id) for source_id in raw_ids)
        selected_ids = kernel["selected_occurrence_ids"]
        selected_benchmark_ids = tuple(
          _benchmark_id(case, source_id) for source_id in selected_ids
        )
        context_ids = {"selected_occurrence_ids": selected_ids}
        result_ids = {
          "retrieved_occurrence_ids": raw_ids,
          "retrieved_session_ids": raw_benchmark_ids,
          "selected_occurrence_ids": selected_ids,
          "selected_session_ids": selected_benchmark_ids,
        }
      else:
        by_id = {
          value.record.record_id: value
          for value in memories_by_case.get(case.question_id, ())
        }
        selected_ids = kernel["selected_memory_ids"]
        episode_raw_ids = tuple(hit.source_id for hit in episode_hits)
        episode_selected_ids = kernel["selected_episode_occurrence_ids"]
        raw_sources = tuple(by_id[record_id].session.source_id for record_id in raw_ids)
        selected_sources = tuple(by_id[record_id].session.source_id for record_id in selected_ids)
        raw_benchmark_ids = _unique(
          tuple(by_id[record_id].session.benchmark_id for record_id in raw_ids)
          + tuple(_benchmark_id(case, source_id) for source_id in episode_raw_ids)
        )
        selected_benchmark_ids = _unique(
          tuple(by_id[record_id].session.benchmark_id for record_id in selected_ids)
          + tuple(_benchmark_id(case, source_id) for source_id in episode_selected_ids)
        )
        context_ids = {
          "selected_memory_ids": selected_ids,
          "selected_source_occurrence_ids": selected_sources,
        }
        result_ids = {
          "retrieved_memory_ids": raw_ids,
          "retrieved_source_occurrence_ids": raw_sources,
          "retrieved_session_ids": raw_benchmark_ids,
          "selected_memory_ids": selected_ids,
          "selected_source_occurrence_ids": selected_sources,
          "selected_session_ids": selected_benchmark_ids,
        }
        if args.episodes:
          episode_fields = {
            "retrieved_episode_occurrence_ids": episode_raw_ids,
            "selected_episode_occurrence_ids": episode_selected_ids,
            "episode_omissions": tuple(
              {
                "source_occurrence_id": session.source_id,
                "reason": "after_cutoff",
              }
              for session in case.sessions
              if session.at > case.asked_at
            ),
          }
          context_ids = {
            **context_ids,
            "selected_episode_occurrence_ids": episode_selected_ids,
          }
          result_ids = {**result_ids, **episode_fields}
      contexts.append(
        {
          "question_id": case.question_id,
          "question": case.question,
          "question_date": case.question_date,
          **context_ids,
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
          **result_ids,
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
    if episode_qdrant is not None:
      episode_qdrant.close()

  config = {
    "retriever": args.retriever,
    "offset": args.offset,
    "limit": len(cases),
    "candidates": args.candidates,
    "top_k": args.top_k,
    "chars": args.chars,
    "lexical_weight": args.lexical_weight,
    "retriever_identity": retriever_identity,
  }
  if artifact is not None:
    config = {**config, "source": "memories", "artifact": artifact.identity}
  if args.episodes:
    config = {**config, "episodes": args.episodes}
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
    "config": config,
    "summary": _summary(results),
    "cases": results,
  }
  run_id = _digest(semantic)
  context_output = None
  if args.contexts is not None:
    context_output = {
      "file": args.contexts.name,
      "sha256": _write_jsonl(
        args.contexts,
        [{**context, "run_id": run_id} for context in contexts],
      ),
    }
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
    "contexts": context_output,
  }
  receipt = {**receipt, "receipt_sha256": _digest(receipt)}
  out = args.out or Path("runs") / f"{args.retriever}-{run_id[:12]}.json"
  _write(out, receipt)
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


def _load_memories(
  args: argparse.Namespace,
  *,
  source: dict[str, Any],
  cases: tuple[Case, ...],
) -> tuple[memory.Artifact | None, dict[str, tuple[Memory, ...]]]:
  import memory

  if args.source == "sessions":
    return None, {}
  try:
    artifact = memory.load(args.artifact, sha256=args.artifact_sha256)
  except memory.MemoryCompileError as exc:
    raise LongMemError(str(exc)) from exc
  if (
    artifact.benchmark_repository != _BENCHMARK_REPOSITORY
    or artifact.benchmark_revision != _BENCHMARK_REVISION
  ):
    raise LongMemError("memory_artifact_benchmark_mismatch")
  if (
    artifact.dataset_repository != source["repository"]
    or artifact.dataset_revision != source["revision"]
    or artifact.dataset_sha256 != source["sha256"]
    or artifact.dataset_size != source["size"]
  ):
    raise LongMemError("memory_artifact_dataset_mismatch")

  by_case = {case.question_id: case for case in cases}
  sessions_by_case = {
    case.question_id: {session.source_id: session for session in case.sessions}
    for case in cases
  }
  all_sources = {
    source_id
    for sessions in sessions_by_case.values()
    for source_id in sessions
  }
  grouped: dict[str, list[Memory]] = {}
  for record in artifact.records:
    case = by_case.get(record.case_id)
    if case is None:
      raise LongMemError("memory_artifact_case_missing")
    session = sessions_by_case[record.case_id].get(record.source.fragment)
    if session is None:
      reason = (
        "memory_artifact_source_scope_mismatch"
        if record.source.fragment in all_sources
        else "memory_artifact_source_missing"
      )
      raise LongMemError(reason)
    if record.source_date != session.date:
      raise LongMemError("memory_artifact_source_date_mismatch")
    grouped.setdefault(record.case_id, []).append(Memory(record=record, session=session))
  return artifact, {
    case_id: tuple(sorted(values, key=lambda value: value.record.record_id))
    for case_id, values in grouped.items()
  }


def _session_entries(case: Case) -> tuple[Entry, ...]:
  return tuple(
    Entry(
      source_id=session.source_id,
      benchmark_id=session.benchmark_id,
      text=session.text,
      at=session.at,
    )
    for session in case.sessions
  )


def _episode_entries(case: Case) -> tuple[Entry, ...]:
  return tuple(entry for entry in _session_entries(case) if entry.at <= case.asked_at)


def _memory_entries(values: tuple[Memory, ...]) -> tuple[Entry, ...]:
  return tuple(
    Entry(
      source_id=value.record.record_id,
      benchmark_id=value.session.benchmark_id,
      text=value.record.text,
      at=value.session.at,
    )
    for value in values
  )


def _retrieve(case: Case, *, retriever: _Retriever, limit: int) -> tuple[Hit, ...]:
  return _retrieve_entries(
    case,
    _session_entries(case),
    retriever=retriever,
    limit=limit,
  )


def _retrieve_entries(
  case: Case,
  entries: tuple[Entry, ...],
  *,
  retriever: _Retriever,
  limit: int,
) -> tuple[Hit, ...]:
  if not entries or retriever == "none":
    return ()
  if retriever == "recent":
    ordered = sorted(enumerate(entries), key=lambda item: (-item[1].at.timestamp(), item[0]))
  elif retriever == "full":
    ordered = list(enumerate(entries))
    limit = len(entries)
    if limit > 100:
      raise LongMemError("full_retriever_route_limit_exceeded")
  elif retriever == "lexical":
    bm25 = BM25Okapi([entry.text.split(" ") for entry in entries])
    scores = bm25.get_scores(case.question.split(" "))
    ordered = [(index, entries[index]) for index in scores.argsort()[::-1]]
  elif retriever == "oracle":
    answer = set(case.answer_session_ids)
    ordered = sorted(enumerate(entries), key=lambda item: (item[1].benchmark_id not in answer, item[0]))
  else:
    raise LongMemError("qdrant_dependency_missing:run_with_uv_script")
  return tuple(Hit(entry.source_id) for _, entry in ordered[:limit])


def _retrieve_episodes(
  case: Case,
  entries: tuple[Entry, ...],
  *,
  retriever: _Retriever,
  limit: int,
) -> tuple[Hit, ...]:
  if retriever == "full":
    return tuple(Hit(entry.source_id) for entry in entries[:limit])
  return _retrieve_entries(case, entries, retriever=retriever, limit=limit)


def _rrf(case: Case, *rankings: Ranking, limit: int) -> tuple[Hit, ...]:
  return _rrf_entries(_session_entries(case), *rankings, limit=limit)


def _rrf_entries(
  entries: tuple[Entry, ...],
  *rankings: Ranking,
  limit: int,
) -> tuple[Hit, ...]:
  scores: dict[str, float] = {}
  paths: dict[str, list[SelectionPath]] = {}
  for ranking in rankings:
    identities = tuple(hit.source_id for hit in ranking.hits)
    if len(identities) != len(set(identities)):
      raise LongMemError("rrf_ranking_identity_duplicated")
    for rank, hit in enumerate(ranking.hits, start=1):
      scores[hit.source_id] = scores.get(hit.source_id, 0.0) + 1 / (60 + rank)
      paths.setdefault(hit.source_id, []).append(
        SelectionPath(lane=ranking.lane, rank=rank, signal="retrieved")
      )
  order = {entry.source_id: index for index, entry in enumerate(entries)}
  if not set(scores) <= set(order):
    raise LongMemError("rrf_result_scope_mismatch")
  source_ids = sorted(scores, key=lambda source_id: (-scores[source_id], order[source_id]))
  return tuple(Hit(source_id, tuple(paths[source_id])) for source_id in source_ids[:limit])


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
  items = tuple(_session_item(by_id[hit.source_id], paths=hit.paths) for hit in hits)
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


def _session_item(
  session: Session,
  *,
  paths: tuple[SelectionPath, ...] = (),
) -> SelectionItem:
  return SelectionItem(
    source="session",
    source_id=session.source_id,
    partition="history",
    kind="session",
    text=session.text,
    content=(
      f"Session Date: {session.date}\nSession Content:\n{session.content}"
      if session.content
      else ""
    ),
    updated_at=session.at,
    revision=session.date,
    source_digest=source_digest(session.content),
    paths=paths,
  )


def _govern_memory(
  case: Case,
  values: tuple[Memory, ...],
  hits: tuple[Hit, ...],
  *,
  episode_hits: tuple[Hit, ...] = (),
  artifact: memory.Artifact,
  source: dict[str, Any],
  retriever: _Retriever,
  top_k: int,
  chars: int,
  lexical_weight: int,
) -> dict[str, Any]:
  if any(value.record.case_id != case.question_id for value in values):
    raise LongMemError("memory_record_scope_mismatch")
  by_id = {value.record.record_id: value for value in values}
  if any(hit.source_id not in by_id for hit in hits):
    raise LongMemError("retrieval_result_scope_mismatch")
  sessions = {
    session.source_id: session
    for session in case.sessions
    if session.at <= case.asked_at
  }
  if any(hit.source_id not in sessions for hit in episode_hits):
    raise LongMemError("episode_retrieval_result_scope_mismatch")
  retained = {}
  reopened = {}
  items = []
  policy = RetentionPolicy()
  for hit in hits:
    value = by_id[hit.source_id]
    decision = retain(
      RetentionState(
        updated_at=value.session.at,
        expires_at=value.record.expires_at,
      ),
      policy=policy,
      now=case.asked_at,
    )
    retained[hit.source_id] = decision
    reopened[hit.source_id] = _reopen(case, value, source=source)
    if reopened[hit.source_id]["decision"] != "current":
      continue
    items.append(
      SelectionItem(
        source="memory",
        source_id=value.record.record_id,
        partition="case",
        kind=value.record.kind,
        text=value.record.text,
        content=value.record.text,
        confidence=value.record.confidence,
        updated_at=value.session.at,
        revision=artifact.artifact_id,
        source_digest=value.record.source.digest,
        omission="retention" if isinstance(decision, Omit) else None,
        paths=hit.paths,
      )
    )
  items.extend(
    _session_item(sessions[hit.source_id], paths=hit.paths)
    for hit in episode_hits
  )
  routes = tuple(
    SelectionRoute(
      source="memory",
      source_id=hit.source_id,
      lane=retriever,
      rank=rank,
      signal=retriever,
    )
    for rank, hit in enumerate(hits, start=1)
    if reopened[hit.source_id]["decision"] == "current"
  ) + tuple(
    SelectionRoute(
      source="session",
      source_id=hit.source_id,
      lane=retriever,
      rank=rank,
      signal=retriever,
    )
    for rank, hit in enumerate(episode_hits, start=1)
  )
  mixed = bool(episode_hits)
  selection = select(
    tuple(items),
    routes=routes,
    query=case.question,
    limits=SelectionLimits(max_items=top_k, max_chars=chars),
    policy=SelectionPolicy(
      partitions=(SelectionPriority(label="case", score=0),)
      + ((SelectionPriority(label="history", score=0),) if mixed else ()),
      kinds=tuple(
        SelectionPriority(label=kind, score=0)
        for kind in sorted({item.kind for item in items})
      ),
      source_order=("memory", "session") if mixed else ("memory",),
      route_order=(retriever,),
      lexical_weight=lexical_weight,
      route_rank_ceiling=100,
    ),
    now=case.asked_at,
    include_paths=True,
  )
  outcomes = tuple(_outcome(outcome, exact=mixed) for outcome in selection.outcomes)
  selected_memory = tuple(
    outcome.candidate.source_id
    for outcome in sorted(selection.selected, key=lambda item: item.rank)
    if outcome.candidate.source == "memory"
  )
  selected_episodes = tuple(
    outcome.candidate.source_id
    for outcome in sorted(selection.selected, key=lambda item: item.rank)
    if outcome.candidate.source == "session"
  )
  if any(reopened[record_id]["decision"] != "current" for record_id in selected_memory):
    raise LongMemError("memory_selected_support_invalid")
  return {
    "selected_memory_ids": selected_memory,
    "selected_episode_occurrence_ids": selected_episodes,
    "content": selection.content,
    "receipt": {
      "content_chars": len(selection.content),
      "content_sha256": hashlib.sha256(selection.content.encode()).hexdigest(),
      "source_count": selection.source_count,
      "included_count": selection.included_count,
      "truncated": selection.truncated,
      "retention": tuple(
        {
          "record_id": hit.source_id,
          "decision": "omit" if isinstance(retained[hit.source_id], Omit) else "retain",
          **(
            {"reason": retained[hit.source_id].reason}
            if isinstance(retained[hit.source_id], Omit)
            else {}
          ),
        }
        for hit in hits
      ),
      "support": tuple(reopened[hit.source_id] for hit in hits),
      "outcomes": outcomes,
    },
  }


def _reopen(
  case: Case,
  value: Memory,
  *,
  source: dict[str, Any],
) -> dict[str, str]:
  if value.record.case_id != case.question_id:
    raise LongMemError("memory_record_reopen_scope_mismatch")
  current = ReopenedSource(
    owner=f"{source['repository']}#{case.question_id}",
    revision=source["revision"],
    fragment=value.session.source_id,
    kind="session",
    digest=source_digest(value.session.content),
    current_revision=source["revision"],
  )
  decision = support(value.record.source, current)
  return {
    "record_id": value.record.record_id,
    "source_ref": value.record.source_ref,
    "source_occurrence_id": value.session.source_id,
    "source_digest": value.record.source.digest,
    "reopened_digest": current.digest,
    "decision": type(decision).__name__.lower(),
  }


def _outcome(outcome: Selected | Omitted, *, exact: bool = False) -> dict[str, Any]:
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
  if exact:
    value = {
      **value,
      "source": candidate.source,
      "revision": candidate.revision,
      "source_digest": candidate.source_digest,
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
    entries_by_case: dict[str, tuple[Entry, ...]],
    *,
    hybrid: bool,
    model: str,
    cache: Path,
    text: str,
    embedder: Any | None = None,
  ) -> None:
    try:
      from fastembed import TextEmbedding
      from qdrant_client import QdrantClient, models
    except ImportError as exc:
      raise LongMemError("qdrant_dependency_missing:run_with_uv_script") from exc

    cache.mkdir(parents=True, exist_ok=True)
    self._models = models
    self._client = QdrantClient(":memory:")
    self.embedder = embedder if embedder is not None else TextEmbedding(model, cache_dir=str(cache))
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
      "dense": _model_identity(self.embedder),
      "fusion": (
        {"algorithm": "rrf", "k": 60, "lexical": _retriever_identity("lexical", text=text)}
        if hybrid
        else None
      ),
      **({"text": text} if text != "user turns" else {}),
    }

    rows = tuple(
      (case.question_id, entry)
      for case in cases
      for entry in entries_by_case[case.question_id]
    )
    self._client.create_collection(
      self._collection,
      vectors_config={
        "dense": models.VectorParams(size=self.embedder.embedding_size, distance=models.Distance.COSINE)
      },
    )
    for start in range(0, len(rows), 256):
      batch = rows[start : start + 256]
      dense_vectors = self.embedder.embed(entry.text for _, entry in batch)
      points = [
        models.PointStruct(
          id=start + offset,
          payload={"question_id": question_id, "source_id": entry.source_id},
          vector={"dense": vector.tolist()},
        )
        for offset, ((question_id, entry), vector) in enumerate(
          zip(batch, dense_vectors, strict=True)
        )
      ]
      self._client.upsert(self._collection, points=points, wait=True)

  def retrieve(
    self,
    case: Case,
    entries: tuple[Entry, ...],
    *,
    limit: int,
  ) -> tuple[Hit, ...]:
    if not entries:
      return ()
    models = self._models
    dense = next(iter(self.embedder.query_embed(case.question))).tolist()
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
    allowed = {entry.source_id for entry in entries}
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
    lexical = _retrieve_entries(case, entries, retriever="lexical", limit=limit)
    return _rrf_entries(
      entries,
      Ranking("qdrant-dense", hits),
      Ranking("lexical", lexical),
      limit=limit,
    )

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
  if args.episodes < 0 or args.episodes > args.top_k:
    raise LongMemError("episode_candidate_limit_invalid")
  if args.episodes and args.source != "memories":
    raise LongMemError("episode_candidates_require_memories")
  if args.source == "memories":
    if args.artifact is None or not args.artifact_sha256:
      raise LongMemError("memory_artifact_identity_required")
  elif args.artifact is not None or args.artifact_sha256 is not None:
    raise LongMemError("memory_artifact_requires_memory_source")
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


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
  return tuple(dict.fromkeys(values))


def _write(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix(f"{path.suffix}.part")
  partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  os.replace(partial, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> str:
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix(f"{path.suffix}.part")
  encoded = "".join(f"{json.dumps(value, sort_keys=True)}\n" for value in values)
  partial.write_text(encoded)
  os.replace(partial, path)
  return hashlib.sha256(encoded.encode()).hexdigest()


def _digest(value: Any) -> str:
  return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _retriever_identity(retriever: _Retriever, *, text: str = "user turns") -> dict[str, Any]:
  if retriever == "lexical":
    return {
      "package": "rank-bm25",
      "version": version("rank-bm25"),
      "numpy": version("numpy"),
      "algorithm": "BM25Okapi",
      "text": text,
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
