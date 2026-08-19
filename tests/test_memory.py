import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import longmem
import memory
from agos_memory.support import source_digest


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixture.json"
EXTRACTOR = ROOT / "tests" / "extractor.jsonl"


def test_compile_emits_exact_admission_and_replacement_lineage(tmp_path: Path) -> None:
  completed, receipt = _run(tmp_path)
  report = json.loads(completed.stdout)
  artifact = receipt["artifact"]
  artifact_body = {key: value for key, value in artifact.items() if key != "artifact_id"}
  receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}

  assert report["artifact_id"] == artifact["artifact_id"]
  assert report["sha256"] == _file_digest(tmp_path / "memory.json")
  assert artifact["artifact_id"] == memory._digest(artifact_body)
  assert receipt["receipt_sha256"] == memory._digest(receipt_body)
  assert artifact["summary"] == {
    "active_records": 3,
    "cases": 2,
    "outcomes": {"accept": 3, "reject": 1, "replace": 1},
    "proposals": 5,
  }

  degree, update = artifact["cases"]
  assert [decision["outcome"] for decision in degree["decisions"]] == ["accept", "reject"]
  assert degree["decisions"][0]["proposal"]["text"] == "The user bought a new bicycle."
  assert degree["decisions"][1]["reason"] == "duplicate"
  old, replacement, event = update["decisions"]
  assert replacement["outcome"] == "replace"
  assert replacement["replaced_record_ids"] == [old["record_id"]]
  assert old["record_id"] not in update["active_record_ids"]
  assert update["active_record_ids"] == sorted((replacement["record_id"], event["record_id"]))
  assert old["source"]["fragment"] == "0:old-color"

  for case in artifact["cases"]:
    for decision in case["decisions"]:
      source = decision["source"]
      source_body = {key: value for key, value in source.items() if key != "ref"}
      assert source["ref"] == f"source:{memory._digest(source_body)}"
      assert decision["proposal"]["source_refs"] == [source["ref"]]
      if decision["outcome"] != "reject":
        decision_body = {
          key: value
          for key, value in decision.items()
          if key not in {"proposal_id", "record_id"}
        }
        assert decision["record_id"] == "memory:" + memory._digest(
          {
            "schema": "agos-memory-lab-record-v1",
            "extractor": artifact["extractor"],
            "decision": decision_body,
          }
        )


def test_identical_inputs_are_byte_identical(tmp_path: Path) -> None:
  _, first = _run(tmp_path, name="first")
  _, second = _run(tmp_path, name="second")

  assert first == second
  assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()


def test_benchmark_labels_do_not_enter_the_memory_artifact(tmp_path: Path) -> None:
  source_case = memory._sources(longmem._load(FIXTURE))[0]
  assert not hasattr(source_case, "question")
  assert not hasattr(source_case, "question_type")
  assert not hasattr(source_case, "answer_session_ids")

  _, baseline = _run(tmp_path, name="baseline")
  values = json.loads(FIXTURE.read_text())
  for value in values:
    value["question"] = "Changed benchmark question"
    value["question_type"] = "changed-type"
    value["answer"] = "Changed gold answer"
    value["answer_session_ids"] = value["haystack_session_ids"][:1]
    for session in value["haystack_sessions"]:
      for turn in session:
        turn.pop("has_answer", None)
  dataset = _write_json(tmp_path / "labels.json", values)

  _, changed = _run(tmp_path, name="labels", dataset=dataset)

  assert changed["artifact"] == baseline["artifact"]
  assert changed["run_id"] != baseline["run_id"]


def test_extractor_proposal_and_source_changes_change_artifact_identity(tmp_path: Path) -> None:
  _, baseline = _run(tmp_path, name="baseline")
  values = _extractor_values()

  extractor_identity = deepcopy(values)
  extractor_identity[0]["config"]["model"] = "synthetic-v2"
  _, changed_extractor = _run(
    tmp_path,
    name="extractor",
    extractor=_write_jsonl(tmp_path / "extractor-identity.jsonl", extractor_identity),
  )

  proposal = deepcopy(values)
  proposal[-1]["text"] = "The user visited the library."
  _, changed_proposal = _run(
    tmp_path,
    name="proposal",
    extractor=_write_jsonl(tmp_path / "proposal.jsonl", proposal),
  )

  dataset_values = json.loads(FIXTURE.read_text())
  dataset_values[0]["haystack_sessions"][0][0]["content"] = "I bought a red bicycle."
  dataset = _write_json(tmp_path / "source.json", dataset_values)
  source = deepcopy(values)
  source[1]["source_digest"] = source_digest(
    "user: I bought a red bicycle.\nassistant: That sounds fun."
  )
  _, changed_source = _run(
    tmp_path,
    name="source",
    dataset=dataset,
    extractor=_write_jsonl(tmp_path / "source.jsonl", source),
  )

  artifact_id = baseline["artifact"]["artifact_id"]
  assert changed_extractor["artifact"]["artifact_id"] != artifact_id
  assert changed_proposal["artifact"]["artifact_id"] != artifact_id
  assert changed_source["artifact"]["artifact_id"] != artifact_id


def test_extractor_line_order_does_not_change_semantic_artifact(tmp_path: Path) -> None:
  _, baseline = _run(tmp_path, name="baseline")
  header, *rows = _extractor_values()
  extractor = _write_jsonl(tmp_path / "reordered.jsonl", [header, *reversed(rows)])

  _, reordered = _run(tmp_path, name="reordered", extractor=extractor)

  assert reordered["artifact"] == baseline["artifact"]
  assert reordered["run_id"] != baseline["run_id"]


@pytest.mark.parametrize(
  ("mutate", "error"),
  (
    (lambda rows: rows[1].update(case_id="unknown"), "extractor_case_missing"),
    (lambda rows: rows[1].update(source_id="99:missing"), "extractor_source_missing"),
    (lambda rows: rows[1].update(source_id="1:new-color"), "extractor_source_scope_mismatch"),
    (
      lambda rows: rows[1].update(
        source_id="2:answer-1",
        session_date="2024/01/03 (Wed) 13:00",
        source_digest="sha256:120e2d671900f6c06b579d17f5852a28147019fa8d5c3f3f5e718ab5f59c8723",
      ),
      "extractor_source_future",
    ),
    (
      lambda rows: rows[1].update(source_digest="sha256:" + "0" * 64),
      "extractor_source_digest_mismatch",
    ),
    (
      lambda rows: rows[1].update(session_date="2024/01/02 (Tue) 12:00"),
      "extractor_source_date_mismatch",
    ),
    (lambda rows: rows[2].update(proposal_id="bicycle-1"), "extractor_proposal_identity_duplicated"),
    (
      lambda rows: rows[2].update(
        source_id="0:noise-1",
        session_date="2024/01/01 (Mon) 12:00",
      ),
      "extractor_proposal_ordinal_duplicated",
    ),
  ),
)
def test_source_and_proposal_boundaries_fail_closed(
  tmp_path: Path,
  mutate: Any,
  error: str,
) -> None:
  values = _extractor_values()
  mutate(values)
  extractor = _write_jsonl(tmp_path / f"{error}.jsonl", values)

  completed, _ = _run(tmp_path, name=error, extractor=extractor, check=False)

  assert completed.returncode != 0
  assert completed.stderr.strip() == error


def test_cross_case_replacement_fails_closed(tmp_path: Path) -> None:
  values = _extractor_values()
  values[-1]["supersedes"] = ["bicycle-1"]
  extractor = _write_jsonl(tmp_path / "cross-case.jsonl", values)

  completed, _ = _run(tmp_path, name="cross-case", extractor=extractor, check=False)

  assert completed.returncode != 0
  assert completed.stderr.strip() == "extractor_supersedes_scope_mismatch"


def test_extractor_cannot_smuggle_benchmark_labels(tmp_path: Path) -> None:
  values = _extractor_values()
  values[1]["answer"] = "gold"
  extractor = _write_jsonl(tmp_path / "gold.jsonl", values)

  completed, _ = _run(tmp_path, name="gold", extractor=extractor, check=False)

  assert completed.returncode != 0
  assert completed.stderr.strip() == "extractor_row_invalid"


def _run(
  tmp_path: Path,
  *,
  name: str = "memory",
  dataset: Path = FIXTURE,
  extractor: Path = EXTRACTOR,
  check: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
  out = tmp_path / f"{name}.json"
  completed = subprocess.run(
    [
      sys.executable,
      "memory.py",
      "--file",
      str(dataset),
      "--sha256",
      _file_digest(dataset),
      "--revision",
      "fixture-v1",
      "--extractor",
      str(extractor),
      "--extractor-sha256",
      _file_digest(extractor),
      "--out",
      str(out),
    ],
    cwd=ROOT,
    check=check,
    capture_output=True,
    text=True,
  )
  return completed, json.loads(out.read_text()) if out.exists() else {}


def _extractor_values() -> list[dict[str, Any]]:
  return [json.loads(line) for line in EXTRACTOR.read_text().splitlines()]


def _write_json(path: Path, value: Any) -> Path:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  return path


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> Path:
  path.write_text("".join(f"{json.dumps(value, sort_keys=True)}\n" for value in values))
  return path


def _file_digest(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()
