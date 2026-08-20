import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import longmem
import memory


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixture.json"
EXTRACTOR = ROOT / "tests" / "extractor.jsonl"
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
  assert receipt["run_id"] == "76bd53e23226824c4cb53456ff2dea094e74c12c863bb89cb833ce10f871849e"
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
  assert context["run_id"] == receipt["run_id"]
  assert receipt["contexts"] == {
    "file": "contexts.jsonl",
    "sha256": hashlib.sha256(contexts.read_bytes()).hexdigest(),
  }
  assert "assistant: Congratulations." in context["context"]
  assert "answer" not in context
  assert receipt_hash == hashlib.sha256(
    json.dumps(receipt, separators=(",", ":"), sort_keys=True).encode()
  ).hexdigest()


def test_memory_run_reopens_source_linked_records_before_context(tmp_path: Path) -> None:
  artifact, artifact_sha256 = _artifact(tmp_path)
  out = tmp_path / "memory-receipt.json"
  contexts = tmp_path / "memory-contexts.jsonl"

  _run(
    "run",
    "--file",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--source",
    "memories",
    "--artifact",
    str(artifact),
    "--artifact-sha256",
    artifact_sha256,
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

  receipt = json.loads(out.read_text())
  context = json.loads(contexts.read_text().splitlines()[1])
  update = receipt["cases"][1]

  assert receipt["config"]["source"] == "memories"
  assert receipt["contexts"] == {
    "file": "memory-contexts.jsonl",
    "sha256": hashlib.sha256(contexts.read_bytes()).hexdigest(),
  }
  assert receipt["config"]["artifact"]["sha256"] == artifact_sha256
  assert receipt["config"]["artifact"]["extractor"]["sha256"] == hashlib.sha256(
    EXTRACTOR.read_bytes()
  ).hexdigest()
  assert receipt["config"]["artifact"]["admission"] == {
    "cases": 2,
    "proposals": 5,
    "outcomes": {"accept": 3, "reject": 1, "replace": 1},
    "reasons": {"duplicate": 1},
  }
  assert "selected_occurrence_ids" not in update
  assert update["selected_session_ids"] == ["noise-2", "new-color"]
  assert update["selected_source_occurrence_ids"] == ["2:noise-2", "1:new-color"]
  assert [item["decision"] for item in update["kernel"]["retention"]] == ["retain", "retain"]
  assert [item["decision"] for item in update["kernel"]["support"]] == ["current", "current"]
  assert [item["record_id"] for item in update["kernel"]["support"]] == update["selected_memory_ids"]
  assert "The user's favorite color is green." in context["context"]
  assert "blue" not in context["context"]
  assert "assistant:" not in context["context"]
  assert "answer" not in context


def test_memory_run_omits_an_expired_record_before_context(tmp_path: Path) -> None:
  rows = [json.loads(line) for line in EXTRACTOR.read_text().splitlines()]
  rows[4]["expires_days"] = 1
  extractor = tmp_path / "expiring.jsonl"
  extractor.write_text("".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows))
  artifact, artifact_sha256 = _artifact(tmp_path, extractor=extractor)
  out = tmp_path / "expired.json"
  contexts = tmp_path / "expired-contexts.jsonl"

  _run(
    "run",
    "--file",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--source",
    "memories",
    "--artifact",
    str(artifact),
    "--artifact-sha256",
    artifact_sha256,
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

  update = json.loads(out.read_text())["cases"][1]
  context = json.loads(contexts.read_text().splitlines()[1])["context"]

  assert update["selected_session_ids"] == ["noise-2"]
  assert {item.get("reason") for item in update["kernel"]["retention"]} == {None, "expired"}
  assert "favorite color" not in context


def test_memory_source_requires_an_exact_matching_artifact(tmp_path: Path) -> None:
  artifact, artifact_sha256 = _artifact(tmp_path)
  base = (
    "run",
    "--file",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--source",
    "memories",
    "--out",
    str(tmp_path / "no.json"),
  )

  missing = _run(*base, check=False)
  wrong_digest = _run(
    *base,
    "--artifact",
    str(artifact),
    "--artifact-sha256",
    "0" * 64,
    check=False,
  )
  wrong_revision = _run(
    "run",
    "--file",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v2",
    "--source",
    "memories",
    "--artifact",
    str(artifact),
    "--artifact-sha256",
    artifact_sha256,
    "--out",
    str(tmp_path / "no.json"),
    check=False,
  )

  assert missing.stderr.strip() == "memory_artifact_identity_required"
  assert wrong_digest.stderr.strip() == "memory_artifact_sha256_mismatch"
  assert wrong_revision.stderr.strip() == "memory_artifact_dataset_mismatch"


def test_memory_support_omits_source_drift_and_records_the_failure(tmp_path: Path) -> None:
  artifact_path, artifact_sha256 = _artifact(tmp_path)
  cases = longmem._load(FIXTURE)
  source = {
    "repository": "custom",
    "revision": "fixture-v1",
    "sha256": FIXTURE_SHA256,
    "size": FIXTURE.stat().st_size,
  }
  args = SimpleNamespace(
    source="memories",
    artifact=artifact_path,
    artifact_sha256=artifact_sha256,
  )
  changed = replace(
    cases[1],
    sessions=tuple(
      replace(session, content="changed")
      if session.source_id == "1:new-color"
      else session
      for session in cases[1].sessions
    ),
  )
  artifact, grouped = longmem._load_memories(
    args,
    source=source,
    cases=(cases[0], changed, cases[2]),
  )
  stale = next(
    value
    for value in grouped["update"]
    if value.session.source_id == "1:new-color"
  )

  result = longmem._govern_memory(
    changed,
    grouped["update"],
    (longmem.Hit(stale.record.record_id),),
    artifact=artifact,
    source=source,
    retriever="lexical",
    top_k=1,
    chars=100,
    lexical_weight=0,
  )

  assert result["selected_memory_ids"] == ()
  assert result["content"] == ""
  assert result["receipt"]["support"] == (
    {
      "record_id": stale.record.record_id,
      "source_ref": stale.record.source_ref,
      "source_occurrence_id": "1:new-color",
      "source_digest": stale.record.source.digest,
      "reopened_digest": longmem.source_digest("changed"),
      "decision": "stale",
    },
  )
  assert result["receipt"]["outcomes"] == ()


def test_memory_governance_rejects_a_record_from_another_case(tmp_path: Path) -> None:
  artifact_path, artifact_sha256 = _artifact(tmp_path)
  artifact = memory.load(artifact_path, sha256=artifact_sha256)
  cases = longmem._load(FIXTURE)
  source = {
    "repository": "custom",
    "revision": "fixture-v1",
    "sha256": FIXTURE_SHA256,
    "size": FIXTURE.stat().st_size,
  }
  args = SimpleNamespace(
    source="memories",
    artifact=artifact_path,
    artifact_sha256=artifact_sha256,
  )
  _, grouped = longmem._load_memories(args, source=source, cases=cases)
  foreign = grouped["update"][0]

  with pytest.raises(longmem.LongMemError, match="^memory_record_scope_mismatch$"):
    longmem._govern_memory(
      cases[0],
      (foreign,),
      (longmem.Hit(foreign.record.record_id),),
      artifact=artifact,
      source=source,
      retriever="lexical",
      top_k=1,
      chars=100,
      lexical_weight=0,
    )


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


def _artifact(tmp_path: Path, *, extractor: Path = EXTRACTOR) -> tuple[Path, str]:
  path = tmp_path / "memory.json"
  subprocess.run(
    [
      sys.executable,
      "memory.py",
      "--file",
      str(FIXTURE),
      "--sha256",
      FIXTURE_SHA256,
      "--revision",
      "fixture-v1",
      "--extractor",
      str(extractor),
      "--extractor-sha256",
      hashlib.sha256(extractor.read_bytes()).hexdigest(),
      "--out",
      str(path),
    ],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
  )
  return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    [sys.executable, "longmem.py", *args],
    cwd=ROOT,
    check=check,
    capture_output=True,
    text=True,
  )
