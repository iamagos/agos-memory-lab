from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any

from agos_memory.select import select
from agos_memory.support import source_digest, support
from agos_memory.types import (
  ReopenedSource,
  SelectionItem,
  SelectionLimits,
  SelectionPolicy,
  SelectionPriority,
  SelectionRoute,
  SourceDependency,
)


def run() -> dict[str, Any]:
  """Prove correction, exact support, and deterministic selection end to end."""

  now = datetime(2026, 1, 1, tzinfo=timezone.utc)
  query = "When does the debt mature?"
  limits = SelectionLimits(max_items=2, max_chars=200)
  old_text = "Debt matures in 2027."
  current_text = "Debt matures in 2029."
  old = SourceDependency(
    owner="document-1",
    revision="version-1",
    fragment="chunk-1",
    kind="raw",
    digest=source_digest(old_text),
  )
  current = SourceDependency(
    owner="document-1",
    revision="version-2",
    fragment="chunk-1",
    kind="raw",
    digest=source_digest(current_text),
  )
  selection = select(
    (
      SelectionItem(
        source="observation",
        source_id="debt-v2",
        partition="deal",
        kind="fact",
        text=current_text,
        content=current_text,
        updated_at=now,
        revision="version-2",
        source_digest=current.digest,
      ),
    ),
    routes=(
      SelectionRoute(
        source="observation",
        source_id="debt-v2",
        lane="source",
        rank=1,
        signal="exact-source",
      ),
    ),
    query=query,
    limits=limits,
    policy=SelectionPolicy(
      partitions=(SelectionPriority(label="deal", score=20),),
      kinds=(SelectionPriority(label="fact", score=10),),
      source_order=("observation",),
      route_order=("source",),
    ),
    now=now,
    include_paths=True,
  )
  selected = tuple(
    {
      "source": outcome.candidate.source,
      "source_id": outcome.candidate.source_id,
      "candidate_rank": outcome.candidate.candidate_rank,
      "rank": outcome.rank,
      "score": outcome.candidate.score,
      "revision": outcome.candidate.revision,
      "text_hash": outcome.candidate.text_hash,
      "paths": tuple(
        {
          "lane": path.lane,
          "rank": path.rank,
          "signal": path.signal,
          "relation": path.relation,
        }
        for path in outcome.candidate.paths
      ),
    }
    for outcome in selection.selected
  )
  body = {
    "schema": "agos-memory-lab-smoke-v1",
    "kernel": version("agos-memory"),
    "case": "source-correction",
    "input": {
      "now": now.isoformat(),
      "query": query,
      "limits": {"max_items": limits.max_items, "max_chars": limits.max_chars},
    },
    "support": {
      "old": type(
        support(
          old,
          ReopenedSource(
            owner=old.owner,
            revision=old.revision,
            fragment=old.fragment,
            kind=old.kind,
            digest=old.digest,
            current_revision=current.revision,
          ),
        )
      ).__name__.lower(),
      "current": type(
        support(
          current,
          ReopenedSource(
            owner=current.owner,
            revision=current.revision,
            fragment=current.fragment,
            kind=current.kind,
            digest=current.digest,
            current_revision=current.revision,
          ),
        )
      ).__name__.lower(),
    },
    "selection": {
      "content": selection.content,
      "source_count": selection.source_count,
      "included_count": selection.included_count,
      "truncated": selection.truncated,
      "selected": selected,
    },
  }
  digest = hashlib.sha256(_canonical(body)).hexdigest()
  return {**body, "sha256": digest}


def main() -> None:
  print(json.dumps(run(), indent=2, sort_keys=True))


def _canonical(value: dict[str, Any]) -> bytes:
  return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


if __name__ == "__main__":
  main()
