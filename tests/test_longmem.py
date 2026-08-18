import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import longmem


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixture.json"
FIXTURE_SHA256 = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def test_lexical_run_emits_a_verified_governed_receipt(tmp_path: Path) -> None:
  out = tmp_path / "receipt.json"
  contexts = tmp_path / "contexts.jsonl"
  completed = _run(
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
    "--top-k",
    "2",
    "--out",
    str(out),
    "--contexts",
    str(contexts),
  )

  report = json.loads(completed.stdout)
  receipt = json.loads(out.read_text())
  receipt_hash = receipt.pop("receipt_sha256")
  context = json.loads(contexts.read_text().splitlines()[0])

  assert receipt["run_id"] == report["run_id"]
  assert receipt["dataset"]["sha256"] == FIXTURE_SHA256
  assert receipt["summary"]["cases"] == 3
  assert receipt["summary"]["eligible"] == 2
  assert receipt["summary"]["ignored_abstention"] == 1
  assert receipt["cases"][0]["retrieved_session_ids"][0] == "answer-1"
  assert receipt["cases"][0]["retrieved_occurrence_ids"] == [
    "2:answer-1",
    "1:noise-1",
    "0:noise-1",
  ]
  assert receipt["cases"][0]["selected_session_ids"][0] == "answer-1"
  assert context["question_id"] == "degree"
  assert "assistant: Congratulations." in context["context"]
  assert "answer" not in context
  assert receipt_hash == hashlib.sha256(
    json.dumps(receipt, separators=(",", ":"), sort_keys=True).encode()
  ).hexdigest()


def test_oracle_retrieval_is_perfect_on_eligible_fixture_cases(tmp_path: Path) -> None:
  out = tmp_path / "oracle.json"
  _run(
    "run",
    "--file",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--retriever",
    "oracle",
    "--candidates",
    "3",
    "--top-k",
    "1",
    "--out",
    str(out),
  )

  summary = json.loads(out.read_text())["summary"]
  assert summary["retrieval"]["recall_all@1"] == 1.0
  assert summary["kernel"]["recall_all@1"] == 1.0


@pytest.mark.parametrize("retriever", ["none", "recent", "full"])
def test_credential_free_baselines_run_on_the_fixture(tmp_path: Path, retriever: str) -> None:
  out = tmp_path / f"{retriever}.json"

  _run(
    "run",
    "--file",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--retriever",
    retriever,
    "--out",
    str(out),
  )

  receipt = json.loads(out.read_text())
  assert receipt["config"]["retriever"] == retriever
  assert receipt["summary"]["cases"] == 3


def test_custom_dataset_requires_exact_identity(tmp_path: Path) -> None:
  completed = _run(
    "run",
    "--file",
    str(FIXTURE),
    "--sha256",
    "0" * 64,
    "--revision",
    "fixture-v1",
    "--out",
    str(tmp_path / "no.json"),
    check=False,
  )

  assert completed.returncode != 0
  assert completed.stderr.strip() == "dataset_sha256_mismatch"


def test_candidate_limit_cannot_exceed_the_kernel_route_bound(tmp_path: Path) -> None:
  completed = _run(
    "run",
    "--file",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--candidates",
    "101",
    "--out",
    str(tmp_path / "no.json"),
    check=False,
  )

  assert completed.returncode != 0
  assert completed.stderr.strip() == "candidate_limit_invalid"


def test_identical_inputs_have_one_semantic_run_identity(tmp_path: Path) -> None:
  reports = []
  receipts = []
  for index in range(2):
    out = tmp_path / f"receipt-{index}.json"
    completed = _run(
      "run",
      "--file",
      str(FIXTURE),
      "--sha256",
      FIXTURE_SHA256,
      "--revision",
      "fixture-v1",
      "--retriever",
      "lexical",
      "--limit",
      "1",
      "--out",
      str(out),
    )
    reports.append(json.loads(completed.stdout))
    receipt = json.loads(out.read_text())
    receipt.pop("measurements")
    receipt.pop("receipt_sha256")
    receipts.append(receipt)

  assert reports[0]["run_id"] == reports[1]["run_id"]
  assert receipts[0] == receipts[1]


def test_qdrant_mode_fails_with_one_exact_setup_instruction(tmp_path: Path) -> None:
  completed = _run(
    "run",
    "--file",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--retriever",
    "qdrant-dense",
    "--limit",
    "1",
    "--out",
    str(tmp_path / "no.json"),
    check=False,
  )

  assert completed.returncode != 0
  assert completed.stderr.strip() == "qdrant_dependency_missing:run_with_uv_script"


def test_governance_rejects_a_retrieval_result_from_another_case() -> None:
  case = longmem._load(FIXTURE)[0]

  with pytest.raises(longmem.LongMemError, match="^retrieval_result_scope_mismatch$"):
    longmem._govern(
      case,
      (longmem.Hit("foreign"),),
      retriever="lexical",
      top_k=1,
      chars=100,
      lexical_weight=0,
    )


def test_retrieval_uses_user_text_but_selection_retains_the_full_session() -> None:
  session = longmem._load(FIXTURE)[0].sessions[2]

  assert session.text == "I graduated with a degree in Business Administration."
  assert session.content.endswith("assistant: Congratulations.")


def test_hybrid_fusion_is_bounded_and_deterministic() -> None:
  case = longmem._load(FIXTURE)[0]
  first, second, third = (longmem.Hit(session.source_id) for session in case.sessions)

  fused = longmem._rrf(case, (first, second), (second, third), limit=3)

  assert fused == (second, first, third)


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    [sys.executable, "longmem.py", *args],
    cwd=ROOT,
    check=check,
    capture_output=True,
    text=True,
  )
