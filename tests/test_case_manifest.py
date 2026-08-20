import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import case_manifest


@dataclass(frozen=True)
class Case:
  question_id: str
  question_type: str
  answer_session_ids: tuple[str, ...] = ()


CASES = (
  Case("a-1", "alpha", ("source",)),
  Case("a-2_abs", "alpha"),
  Case("a-3", "alpha", ("source",)),
  Case("b-1", "beta", ("source",)),
  Case("b-2", "beta", ("source",)),
  Case("b-3", "beta", ("source",)),
)
BENCHMARK = {"repository": "benchmark", "revision": "benchmark-v1"}
DATASET = {
  "repository": "dataset",
  "revision": "dataset-v1",
  "path": Path("data.json"),
  "sha256": "a" * 64,
  "size": 10,
}


def test_manifest_is_deterministic_stratified_and_source_bound(tmp_path: Path) -> None:
  value = case_manifest.build(
    CASES,
    benchmark=BENCHMARK,
    dataset=DATASET,
    revision="sample-v1",
    seed="seed-v1",
    cases_per_type=2,
    abstention_per_type=1,
  )
  assert value == case_manifest.build(
    CASES,
    benchmark=BENCHMARK,
    dataset=DATASET,
    revision="sample-v1",
    seed="seed-v1",
    cases_per_type=2,
    abstention_per_type=1,
  )
  path = tmp_path / "manifest.json"
  path.write_bytes((json.dumps(value, sort_keys=True) + "\n").encode("utf-8"))

  selected, identity = case_manifest.select(
    path,
    CASES,
    benchmark=BENCHMARK,
    dataset=DATASET,
  )

  assert len(selected) == 4
  assert {case.question_type for case in selected} == {"alpha", "beta"}
  assert sum(case.question_id.endswith("_abs") for case in selected) == 1
  assert identity["question_types"] == {"alpha": 2, "beta": 2}
  assert identity["abstention_by_type"] == {"alpha": 1, "beta": 0}


def test_manifest_rejects_a_different_dataset_identity(tmp_path: Path) -> None:
  value = case_manifest.build(
    CASES,
    benchmark=BENCHMARK,
    dataset=DATASET,
    revision="sample-v1",
    seed="seed-v1",
    cases_per_type=2,
    abstention_per_type=0,
  )
  path = tmp_path / "manifest.json"
  path.write_bytes(json.dumps(value).encode("utf-8"))

  with pytest.raises(case_manifest.ManifestError, match="^case_manifest_source_mismatch$"):
    case_manifest.select(
      path,
      CASES,
      benchmark=BENCHMARK,
      dataset={**DATASET, "revision": "dataset-v2"},
    )
