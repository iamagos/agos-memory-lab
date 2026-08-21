import hashlib
import json
from pathlib import Path

import probe


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixture.json"
FIXTURE_SHA256 = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
RUN_ID = "a" * 64


def test_check_operand_partitions_answerable_contexts(tmp_path: Path) -> None:
  contexts = tmp_path / "contexts.jsonl"
  _jsonl(
    contexts,
    [
      _context("degree", "Business Administration appears here."),
      _context("update", "Only the old color blue appears here."),
      _context("missing_abs", "Dinner was good."),
    ],
  )
  args = _args(
    "check-operand",
    "--contexts",
    str(contexts),
    "--references",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--out",
    str(tmp_path / "operand.json"),
  )

  receipt = probe._check_operand(args)

  assert receipt["summary"] == {
    "cases": 3,
    "answerable": 2,
    "abstention": 1,
    "reader_testable": 1,
    "not_reader_testable": 1,
  }
  assert [case["reader_testability"] for case in receipt["cases"]] == [
    "reader-testable",
    "not-reader-testable",
    "not-applicable",
  ]


def test_answerability_reports_literal_bm25_score_signals(tmp_path: Path) -> None:
  retrieval = tmp_path / "retrieval.json"
  retrieval.write_text(
    json.dumps(
      {
        "schema": "agos-memory-lab-longmem-v1",
        "run_id": RUN_ID,
        "dataset": _dataset(),
        "config": {"retriever": "lexical", "candidates": 3, "top_k": 2},
        "cases": [
          {"question_id": "degree", "question_type": "single-session-user", "abstention": False},
          {"question_id": "update", "question_type": "knowledge-update", "abstention": False},
          {"question_id": "missing_abs", "question_type": "single-session-user", "abstention": True},
        ],
      }
    )
  )
  args = _args(
    "answerability",
    "--receipt",
    str(retrieval),
    "--references",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--out",
    str(tmp_path / "answerability.json"),
  )

  receipt = probe._answerability(args)

  assert receipt["summary"]["answerable"] == 2
  assert receipt["summary"]["abstention"] == 1
  assert receipt["summary"]["auroc"]["top1_score"] is not None
  assert all(case["top1_score"] is not None for case in receipt["cases"])


def test_code_failures_assigns_exactly_one_mechanical_label(tmp_path: Path) -> None:
  retrieval = tmp_path / "retrieval.json"
  contexts = tmp_path / "contexts.jsonl"
  hypotheses = tmp_path / "hypotheses.jsonl"
  evaluations = tmp_path / "evaluations.jsonl"
  retrieval.write_text(
    json.dumps(
      {
        "schema": "agos-memory-lab-longmem-v1",
        "run_id": RUN_ID,
        "dataset": _dataset(),
        "cases": [
          {"question_id": "degree", "retrieved_occurrence_ids": ["2:answer-1"]},
          {"question_id": "update", "retrieved_occurrence_ids": ["1:new-color"]},
          {"question_id": "missing_abs", "retrieved_occurrence_ids": ["0:noise-3"]},
        ],
      }
    )
  )
  _jsonl(
    contexts,
    [
      _context("degree", "The degree was Business Administration."),
      _context("update", "The old color was blue."),
      _context("missing_abs", "Dinner was good."),
    ],
  )
  _jsonl(
    hypotheses,
    [
      {"question_id": "degree", "hypothesis": "Business Administration and Law."},
      {"question_id": "update", "hypothesis": "Blue."},
      {"question_id": "missing_abs", "hypothesis": "In the closet."},
    ],
  )
  _jsonl(
    evaluations,
    [
      {"question_id": "degree", "official_label": False},
      {"question_id": "update", "official_label": False},
      {"question_id": "missing_abs", "official_label": False},
    ],
  )
  args = _args(
    "code-failures",
    "--receipt",
    str(retrieval),
    "--contexts",
    str(contexts),
    "--hypotheses",
    str(hypotheses),
    "--evaluations",
    str(evaluations),
    "--references",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--out",
    str(tmp_path / "failures.json"),
  )

  receipt = probe._code_failures(args)

  assert [case["label"] for case in receipt["cases"]] == [
    "reader-hedge",
    "lost-in-packing",
    "failed-abstention",
  ]
  assert sum(receipt["summary"]["labels"].values()) == receipt["summary"]["judged_wrong"] == 3


def _args(*values: str):
  return probe._parser().parse_args(values)


def _context(question_id: str, context: str) -> dict[str, str]:
  return {
    "run_id": RUN_ID,
    "question_id": question_id,
    "question": "Question?",
    "question_date": "2024/01/03 (Wed) 12:00",
    "context": context,
    "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
  }


def _dataset() -> dict[str, object]:
  return {
    "repository": "custom",
    "revision": "fixture-v1",
    "file": FIXTURE.name,
    "sha256": FIXTURE_SHA256,
    "size": FIXTURE.stat().st_size,
  }


def _jsonl(path: Path, values: list[dict[str, object]]) -> None:
  path.write_text("".join(f"{json.dumps(value)}\n" for value in values))
