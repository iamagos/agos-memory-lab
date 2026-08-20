import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import compare


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixture.json"
EXTRACTOR = ROOT / "tests" / "extractor.jsonl"
FIXTURE_SHA256 = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
EXTRACTOR_SHA256 = hashlib.sha256(EXTRACTOR.read_bytes()).hexdigest()


def test_comparison_reports_exact_retrieval_and_memory_artifact_evidence(tmp_path: Path) -> None:
  sessions, memories = _runs(tmp_path)
  args = _args(sessions, memories, out=tmp_path / "comparison.json")

  receipt = compare._compare(args)
  compare._write(args.out, receipt)

  assert receipt["contract"]["cases"] == ["degree", "update", "missing_abs"]
  assert receipt["contract"]["reader"] is None
  assert receipt["contract"]["judge"] is None
  assert receipt["sessions"]["artifact"] is None
  assert receipt["memories"]["artifact"]["extractor"]["sha256"] == EXTRACTOR_SHA256
  assert receipt["memories"]["artifact"]["admission"] == {
    "cases": 2,
    "proposals": 5,
    "outcomes": {"accept": 3, "reject": 1, "replace": 1},
    "reasons": {"duplicate": 1},
  }
  assert receipt["memories"]["longmem"]["support"] == {
    "checked": 3,
    "failures": 0,
    "decisions": {"current": 3, "missing": 0, "replaced": 0, "stale": 0},
  }
  assert receipt["delta"]["direction"] == "memories_minus_sessions"
  assert receipt["sessions"]["reader"] is None
  assert receipt["memories"]["economics"] is None
  assert receipt["receipt_sha256"] == compare._digest(
    {key: value for key, value in receipt.items() if key != "receipt_sha256"}
  )

  renamed_sessions = tmp_path / "renamed-sessions.json"
  renamed_memories = tmp_path / "renamed-memories.json"
  renamed_sessions.write_bytes(sessions.read_bytes())
  renamed_memories.write_bytes(memories.read_bytes())
  renamed = compare._compare(
    _args(renamed_sessions, renamed_memories, out=tmp_path / "renamed.json")
  )

  assert renamed == receipt


def test_comparison_joins_reader_judge_economics_and_qa(tmp_path: Path) -> None:
  sessions, memories = _runs(tmp_path)
  session_read, session_judge = _qa_pair(tmp_path, "sessions", sessions, accuracy=0.5)
  memory_read, memory_judge = _qa_pair(tmp_path, "memories", memories, accuracy=0.75)

  receipt = compare._compare(
    _args(
      sessions,
      memories,
      out=tmp_path / "comparison.json",
      session_read=session_read,
      memory_read=memory_read,
      session_judge=session_judge,
      memory_judge=memory_judge,
    )
  )

  assert receipt["sessions"]["economics"] == {
    "calls": 6,
    "usage_missing": 0,
    "input_tokens": 15,
    "output_tokens": 3,
    "latency_seconds": 1.5,
    "estimated_cost_usd": 0.15,
    "reserved_cost_usd": 0.3,
  }
  assert receipt["memories"]["economics"]["input_tokens"] == 13
  assert receipt["memories"]["judge"]["summary"]["accuracy"] == 0.75
  assert receipt["delta"]["qa_accuracy"] == 0.25
  assert receipt["delta"]["model_calls"] == 0
  assert receipt["delta"]["model_tokens"] == -2
  assert receipt["delta"]["estimated_cost_usd"] == -0.02


def test_comparison_rejects_unfair_controls_and_unpaired_receipts(tmp_path: Path) -> None:
  sessions, memories = _runs(tmp_path, memory_top_k=1)

  with pytest.raises(compare.CompareError, match="^comparison_retrieval_contract_mismatch$"):
    compare._compare(_args(sessions, memories, out=tmp_path / "no.json"))

  fair_sessions, fair_memories = _runs(tmp_path / "fair")
  session_read, _ = _qa_pair(tmp_path, "sessions-only", fair_sessions, accuracy=0.5)
  with pytest.raises(compare.CompareError, match="^comparison_reader_pair_required$"):
    compare._compare(
      _args(
        fair_sessions,
        fair_memories,
        out=tmp_path / "unpaired.json",
        session_read=session_read,
      )
    )


def test_comparison_rejects_a_tampered_receipt(tmp_path: Path) -> None:
  sessions, memories = _runs(tmp_path)
  value = json.loads(sessions.read_text())
  value["summary"]["selection"]["mean_content_chars"] += 1
  sessions.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

  with pytest.raises(compare.CompareError, match="^sessions_receipt_invalid_digest_mismatch$"):
    compare._compare(_args(sessions, memories, out=tmp_path / "no.json"))


def test_comparison_rejects_different_reader_requests(tmp_path: Path) -> None:
  sessions, memories = _runs(tmp_path)
  session_read, _ = _qa_pair(tmp_path, "sessions", sessions, accuracy=0.5)
  memory_read, _ = _qa_pair(
    tmp_path,
    "memories",
    memories,
    accuracy=0.5,
    model="different-model",
  )

  with pytest.raises(compare.CompareError, match="^comparison_reader_contract_mismatch$"):
    compare._compare(
      _args(
        sessions,
        memories,
        out=tmp_path / "no.json",
        session_read=session_read,
        memory_read=memory_read,
      )
    )


def test_comparison_rejects_a_reader_bound_to_different_contexts(tmp_path: Path) -> None:
  sessions, memories = _runs(tmp_path)
  session_read, _ = _qa_pair(tmp_path, "sessions", sessions, accuracy=0.5)
  memory_read, _ = _qa_pair(tmp_path, "memories", memories, accuracy=0.5)
  value = json.loads(memory_read.read_text())
  value["source"]["sha256"] = _digest_text("different-contexts")
  _resign(value)
  _write(memory_read, value)

  with pytest.raises(compare.CompareError, match="^comparison_reader_source_mismatch$"):
    compare._compare(
      _args(
        sessions,
        memories,
        out=tmp_path / "no.json",
        session_read=session_read,
        memory_read=memory_read,
      )
    )


def test_comparison_rejects_an_unexpected_retrieval_text(tmp_path: Path) -> None:
  _, memories = _runs(tmp_path)
  config = json.loads(memories.read_text())["config"]
  config["retriever_identity"]["text"] = "user turns"

  with pytest.raises(compare.CompareError, match="^comparison_retrieval_text_invalid$"):
    compare._retrieval_contract(config, source="memories")


def _runs(tmp_path: Path, *, memory_top_k: int = 2) -> tuple[Path, Path]:
  tmp_path.mkdir(parents=True, exist_ok=True)
  artifact = tmp_path / "memory.json"
  _command(
    "memory.py",
    "--file",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--extractor",
    str(EXTRACTOR),
    "--extractor-sha256",
    EXTRACTOR_SHA256,
    "--out",
    str(artifact),
  )
  artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
  sessions = tmp_path / "sessions.json"
  memories = tmp_path / "memories.json"
  common = (
    "run",
    "--file",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--retriever",
    "lexical",
    "--candidates",
    "3",
  )
  _command(
    "longmem.py",
    *common,
    "--top-k",
    "2",
    "--out",
    str(sessions),
    "--contexts",
    str(tmp_path / "sessions-contexts.jsonl"),
  )
  _command(
    "longmem.py",
    *common,
    "--source",
    "memories",
    "--artifact",
    str(artifact),
    "--artifact-sha256",
    artifact_sha256,
    "--top-k",
    str(memory_top_k),
    "--out",
    str(memories),
    "--contexts",
    str(tmp_path / "memories-contexts.jsonl"),
  )
  return sessions, memories


def _qa_pair(
  tmp_path: Path,
  lane: str,
  longmem_path: Path,
  *,
  accuracy: float,
  model: str = "fixture-model",
) -> tuple[Path, Path]:
  longmem = json.loads(longmem_path.read_text())
  cases = [case["question_id"] for case in longmem["cases"]]
  config = {
    "request": {
      "provider": "openai",
      "base_url": "https://api.openai.com/v1",
      "api_version": None,
      "model": model,
      "temperature": None,
      "max_tokens": 100,
      "adapter": {"api": "chat-completions", "openai": "3.3.0", "pydantic_ai": "2.21.0"},
    },
    "execution": {
      "timeout": 120.0,
      "concurrency": 1,
      "retries": 0,
      "input_cost_per_million": 1.0,
      "output_cost_per_million": 2.0,
      "max_cost_usd": 1.0,
    },
  }
  read_output_sha256 = _digest_text(f"{lane}:hypotheses")
  read = _qa_receipt(
    kind="read",
    benchmark_revision=longmem["benchmark"]["revision"],
    source={
      "path": longmem["contexts"]["file"],
      "sha256": longmem["contexts"]["sha256"],
      "run_id": longmem["run_id"],
    },
    config=config,
    cases=cases,
    output_sha256=read_output_sha256,
    summary=_usage(cases, memory=lane == "memories"),
  )
  judge_summary = {
    **_usage(cases, judge=True),
    "accuracy": accuracy,
    "task_accuracy": accuracy,
    "abstention_accuracy": accuracy,
    "strict_parse_failures": 0,
    "strict_parse_disagreements": 0,
    "by_type": {},
  }
  scores = {
    key: judge_summary[key]
    for key in (
      "accuracy",
      "task_accuracy",
      "abstention_accuracy",
      "strict_parse_failures",
      "strict_parse_disagreements",
      "by_type",
    )
  }
  judge = _qa_receipt(
    kind="judge",
    benchmark_revision=longmem["benchmark"]["revision"],
    source={
      "hypotheses": {"path": "hypotheses.jsonl", "sha256": read_output_sha256},
      "references": longmem["dataset"],
    },
    config=config,
    cases=cases,
    output_sha256=_digest_text(f"{lane}:evaluation"),
    summary=judge_summary,
    scores=scores,
  )
  read_path = tmp_path / f"{lane}-read.json"
  judge_path = tmp_path / f"{lane}-judge.json"
  _write(read_path, read)
  _write(judge_path, judge)
  return read_path, judge_path


def _qa_receipt(
  *,
  kind: str,
  benchmark_revision: str,
  source: dict[str, Any],
  config: dict[str, Any],
  cases: list[str],
  output_sha256: str,
  summary: dict[str, Any],
  scores: dict[str, Any] | None = None,
) -> dict[str, Any]:
  semantic = {
    "schema": f"agos-memory-lab-{kind}-v4",
    "benchmark_revision": benchmark_revision,
    "source": source,
    "config": config,
    "prompt_revision": "longmem-direct-v1" if kind == "read" else f"longmem-official-{benchmark_revision}",
    "cases": [
      {
        "question_id": question_id,
        "request_id": _digest_text(f"{kind}:{question_id}:request"),
        "result_sha256": _digest_text(f"{kind}:{question_id}:result"),
        "response_model": "fixture-model",
      }
      for question_id in cases
    ],
  }
  if scores is not None:
    semantic["scores"] = scores
  receipt = {
    **semantic,
    "run_id": compare._digest(semantic),
    "summary": summary,
    "output": {"file": f"{kind}.jsonl", "sha256": output_sha256},
  }
  return {**receipt, "receipt_sha256": compare._digest(receipt)}


def _usage(cases: list[str], *, memory: bool = False, judge: bool = False) -> dict[str, Any]:
  if judge:
    return {
      "cases": len(cases),
      "input_tokens": 5,
      "output_tokens": 1,
      "usage_missing": 0,
      "estimated_cost_usd": 0.05,
      "reserved_cost_usd": 0.1,
      "latency_seconds": 0.5,
    }
  return {
    "cases": len(cases),
    "input_tokens": 8 if memory else 10,
    "output_tokens": 2,
    "usage_missing": 0,
    "estimated_cost_usd": 0.08 if memory else 0.1,
    "reserved_cost_usd": 0.2,
    "latency_seconds": 0.8 if memory else 1.0,
  }


def _args(
  sessions: Path,
  memories: Path,
  *,
  out: Path,
  session_read: Path | None = None,
  memory_read: Path | None = None,
  session_judge: Path | None = None,
  memory_judge: Path | None = None,
) -> Any:
  values = ["--sessions", str(sessions), "--memories", str(memories), "--out", str(out)]
  for name, path in (
    ("--sessions-read", session_read),
    ("--memories-read", memory_read),
    ("--sessions-judge", session_judge),
    ("--memories-judge", memory_judge),
  ):
    if path is not None:
      values.extend((name, str(path)))
  return compare._parser().parse_args(values)


def _command(program: str, *args: str) -> None:
  subprocess.run(
    [sys.executable, program, *args],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
  )


def _digest_text(value: str) -> str:
  return hashlib.sha256(value.encode()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _resign(value: dict[str, Any]) -> None:
  semantic_keys = {"schema", "benchmark_revision", "source", "config", "prompt_revision", "cases"}
  if "scores" in value:
    semantic_keys.add("scores")
  value["run_id"] = compare._digest({key: value[key] for key in semantic_keys})
  value["receipt_sha256"] = compare._digest(
    {key: item for key, item in value.items() if key != "receipt_sha256"}
  )
