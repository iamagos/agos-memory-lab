import hashlib
import json
from pathlib import Path

import pytest

import extract
import longmem
import memory


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixture.json"
FIXTURE_SHA256 = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def test_extractor_checkpoints_resumes_and_compiles_without_labels(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  out = tmp_path / "extractor.jsonl"
  args = _args(out, "--limit", "1", "--input-cost", "1.25", "--output-cost", "10", "--max-cost", "1")
  calls: list[str] = []

  def chat(prompt: str, **_: object) -> extract.chat_model.ModelResult[extract.Batch]:
    calls.append(prompt)
    text = (
      "The user earned a degree in Business Administration."
      if "Business Administration" in prompt
      else "The user bought a new bicycle."
    )
    return extract.chat_model.ModelResult(
      extract.Batch(memories=(text,)),
      "gpt-5-2025-08-07",
      100,
      20,
      120,
    )

  monkeypatch.setattr(extract, "_chat", chat)
  monkeypatch.setenv("TEST_EXTRACT_KEY", "secret")
  first = extract._extract(args)
  first_bytes = out.read_bytes()
  jobs = _jobs(args)
  extract._begin(extract._pending_path(out), jobs[0])
  monkeypatch.delenv("TEST_EXTRACT_KEY")
  second = extract._extract(args)

  values = _jsonl(out)
  state = extract._state_path(out).read_text()
  assert len(calls) == 2
  assert first_bytes == out.read_bytes()
  assert first["run_id"] == second["run_id"]
  assert first["summary"]["sources"] == 2
  assert first["summary"]["proposals"] == 2
  assert first["summary"]["input_tokens"] == 200
  assert first["summary"]["output_tokens"] == 40
  assert first["window"]["future_sources_omitted"] == 1
  assert values[0]["config"]["mode"] == "live-additive"
  assert values[0]["config"]["availability"] == "causal"
  assert values[0]["config"]["request"]["model"] == "gpt-5"
  assert values[0]["config"]["response_models"] == ["gpt-5-2025-08-07"]
  assert all(value["kind"] == "fact" for value in values[1:])
  assert all(value["confidence"] == 1.0 for value in values[1:])
  assert all(value["supersedes"] == [] for value in values[1:])
  assert all("question" not in value and "answer" not in value for value in values)
  assert "secret" not in state
  assert not extract._pending_path(out).exists()

  frozen = memory._load_extractor(out, sha256=hashlib.sha256(out.read_bytes()).hexdigest())
  artifact = memory._compile(
    memory._sources(longmem._load(FIXTURE)),
    dataset_repository="fixture",
    dataset_revision="fixture-v1",
    benchmark_repository=longmem._BENCHMARK_REPOSITORY,
    benchmark_revision=longmem._BENCHMARK_REVISION,
    extractor=frozen,
  )
  assert artifact["summary"] == {
    "cases": 1,
    "proposals": 2,
    "outcomes": {"accept": 1, "reject": 1, "replace": 0},
    "active_records": 1,
  }


def test_extraction_jobs_ignore_question_answer_and_gold_labels(tmp_path: Path) -> None:
  changed = json.loads(FIXTURE.read_text())
  for case in changed:
    case["question"] = "changed benchmark question"
    case["answer"] = "changed gold answer"
    case["answer_session_ids"] = []
    for session in case["haystack_sessions"]:
      for turn in session:
        turn["has_answer"] = not turn.get("has_answer", False)
  path = tmp_path / "changed.json"
  path.write_text(json.dumps(changed))
  config = _config()
  source = {"repository": "fixture", "revision": "fixture-v1"}

  original = extract._jobs(
    memory._sources(longmem._load(FIXTURE)),
    source=source,
    config=config,
  )
  altered = extract._jobs(
    memory._sources(longmem._load(path)),
    source=source,
    config=config,
  )

  assert [(job.identity, job.prompt) for job in original] == [
    (job.identity, job.prompt) for job in altered
  ]
  assert "changed benchmark question" not in "".join(job.prompt for job in altered)
  assert "changed gold answer" not in "".join(job.prompt for job in altered)


def test_output_schema_changes_request_identity(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  args = _args(Path("extractor.jsonl"), "--limit", "1")
  original = _jobs(args)

  monkeypatch.setattr(extract, "_output_schema_sha256", lambda: "0" * 64)
  changed = _jobs(args)

  assert [job.request_id for job in original] != [job.request_id for job in changed]


def test_extraction_cost_cap_blocks_before_key_or_call(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls = 0

  def chat(*_: object, **__: object) -> extract.chat_model.ModelResult[extract.Batch]:
    nonlocal calls
    calls += 1
    raise AssertionError

  monkeypatch.setattr(extract, "_chat", chat)
  args = _args(
    tmp_path / "extractor.jsonl",
    "--limit",
    "1",
    "--input-cost",
    "1.25",
    "--output-cost",
    "10",
    "--max-cost",
    "0.000001",
  )

  with pytest.raises(extract.bounded.CallError, match="^chat_cost_cap_reached$"):
    extract._extract(args)

  assert calls == 0


def test_plan_reports_exact_calls_without_credentials_or_writes(tmp_path: Path) -> None:
  args = extract._parser().parse_args(
    [
      "--plan",
      "--file",
      str(FIXTURE),
      "--sha256",
      FIXTURE_SHA256,
      "--revision",
      "fixture-v1",
      "--availability",
      "causal",
      "--limit",
      "1",
      "--model",
      "extractor-v1",
      "--reasoning-effort",
      "minimal",
      "--input-cost",
      "1",
      "--output-cost",
      "2",
      "--max-cost",
      "1",
    ]
  )

  plan = extract._plan(args)

  assert plan["schema"] == "agos-memory-lab-extraction-plan-v1"
  assert plan["window"] == {
    "offset": 0,
    "cases": 1,
    "sources": 2,
    "availability": "causal",
    "future_sources_omitted": 1,
  }
  assert plan["reserved_cost_usd"] > 0
  assert plan["fits_cost_cap"] is True
  assert plan["config"]["request"]["reasoning_effort"] == "minimal"
  assert not tuple(tmp_path.iterdir())


def test_model_output_is_only_bounded_memory_text() -> None:
  schema = extract.Batch.model_json_schema()
  memories = schema["properties"]["memories"]

  assert memories["maxItems"] == 32
  assert memories["items"] == {"type": "string"}
  assert "kind" not in json.dumps(schema)
  assert "confidence" not in json.dumps(schema)
  assert extract.Batch(memories=("  The user   likes tea.  ",)).memories == (
    "The user likes tea.",
  )
  assert extract.Batch(memories=(" ", "The user likes tea." * 100)).memories == (
    "The user likes tea." * 100,
  )
  with pytest.raises(ValueError):
    extract.Batch(memories=("x",) * 33)


def test_specific_assistant_information_is_an_explicit_extraction_source() -> None:
  prompt = extract._prompt(
    memory.Source(
      source_id="source-1",
      date="2023/05/23 (Tue) 07:14",
      at=longmem._date("2023/05/23 (Tue) 07:14"),
      content=(
        "user: What are examples of two-factor authentication?\n"
        "assistant: Biometric authentication and one-time passwords are examples."
      ),
    )
  )

  assert extract._PROMPT_REVISION == "longmem-additive-v3"
  assert "specific recommendations, instructions, solutions, researched facts" in prompt
  assert "even when they are not personal facts about the user" in prompt
  assert "Preserve who supplied the information" in prompt
  assert "Preserve concrete examples, names, numbers, dates, and alternatives" in prompt
  assert "Biometric authentication and one-time passwords" in prompt


def test_released_availability_includes_the_complete_benchmark_haystack() -> None:
  args = _args(Path("extractor.jsonl"), "--limit", "1", "--availability", "released")
  jobs = _jobs(args)

  assert extract._parser().get_default("availability") == "released"
  assert args.availability == "released"
  assert [job.source_id for job in jobs] == ["0:noise-1", "1:noise-1", "2:answer-1"]
  assert extract._future_omitted(
    extract._window(memory._sources(longmem._load(FIXTURE)), offset=0, limit=1),
    availability="released",
  ) == 0


def test_unknown_extraction_outcome_blocks_another_call(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  out = tmp_path / "extractor.jsonl"
  args = _args(out, "--limit", "1")
  job = _jobs(args)[0]
  extract._begin(extract._pending_path(out), job)
  calls = 0

  def chat(*_: object, **__: object) -> extract.chat_model.ModelResult[extract.Batch]:
    nonlocal calls
    calls += 1
    raise AssertionError

  monkeypatch.setattr(extract, "_chat", chat)

  with pytest.raises(extract.ExtractError, match="^extraction_request_outcome_unknown$"):
    extract._extract(args)

  assert calls == 0


@pytest.mark.parametrize(
  "change",
  (
    ("--model", "different-model"),
    ("--reasoning-effort", "minimal"),
  ),
)
def test_changed_request_cannot_reuse_extraction_state(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  change: tuple[str, str],
) -> None:
  out = tmp_path / "extractor.jsonl"
  first = _args(out, "--limit", "1")
  monkeypatch.setenv("TEST_EXTRACT_KEY", "secret")
  monkeypatch.setattr(
    extract,
    "_chat",
    lambda *_args, **_kwargs: extract.chat_model.ModelResult(
      extract.Batch(memories=()),
      "served-model",
      10,
      1,
      11,
    ),
  )
  extract._extract(first)
  changed = _args(out, "--limit", "1", *change)

  with pytest.raises(extract.ExtractError, match="^extraction_resume_identity_mismatch$"):
    extract._extract(changed)


def test_changed_output_schema_cannot_reuse_extraction_state(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  out = tmp_path / "extractor.jsonl"
  args = _args(out, "--limit", "1")
  monkeypatch.setenv("TEST_EXTRACT_KEY", "secret")
  monkeypatch.setattr(
    extract,
    "_chat",
    lambda *_args, **_kwargs: extract.chat_model.ModelResult(
      extract.Batch(memories=()),
      "served-model",
      10,
      1,
      11,
    ),
  )
  extract._extract(args)

  monkeypatch.setattr(extract, "_output_schema_sha256", lambda: "0" * 64)

  with pytest.raises(extract.ExtractError, match="^extraction_resume_identity_mismatch$"):
    extract._extract(args)


def _args(out: Path, *extra: str) -> object:
  values = [
    "--file",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--availability",
    "causal",
    "--out",
    str(out),
    "--model",
    "gpt-5",
    "--api-key-env",
    "TEST_EXTRACT_KEY",
    *extra,
  ]
  return extract._parser().parse_args(values)


def _config() -> extract.bounded.Config:
  return extract.bounded.Config(
    provider="openai",
    provider_id="openai",
    base_url="https://api.openai.com/v1",
    api_version=None,
    model="gpt-5",
    temperature=None,
    reasoning_effort=None,
    max_tokens=1_000,
    max_tokens_field="max_completion_tokens",
    timeout=120,
    input_cost=0,
    output_cost=0,
    max_cost=0,
  )


def _jobs(args: object) -> tuple[extract.Job, ...]:
  config = extract._config(args)
  source = longmem._source(args)
  cases = extract._window(
    memory._sources(longmem._load(FIXTURE)),
    offset=args.offset,
    limit=args.limit,
  )
  return extract._jobs(
    cases,
    source=source,
    config=config,
    availability=args.availability,
  )


def _jsonl(path: Path) -> list[dict[str, object]]:
  return [json.loads(line) for line in path.read_text().splitlines()]
