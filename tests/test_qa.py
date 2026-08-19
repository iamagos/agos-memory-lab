import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import qa


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixture.json"
FIXTURE_SHA256 = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
RUN_ID = "a" * 64


def test_reader_checkpoints_and_exact_resume_makes_no_second_call(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  contexts = tmp_path / "contexts.jsonl"
  out = tmp_path / "hypotheses.jsonl"
  _write_jsonl(contexts, [_context("degree", "What degree?", "Business Administration")])
  args = _args("read", "--contexts", str(contexts), "--out", str(out), "--model", "reader-v1")
  calls: list[str] = []

  def chat(prompt: str, **_: object) -> qa.ChatResult:
    calls.append(prompt)
    return qa.ChatResult("Business Administration.", "reader-v1", 20, 3, 23)

  monkeypatch.setattr(qa, "_chat", chat)
  monkeypatch.setenv("OPENAI_API_KEY", "secret")

  qa._read(args)
  first = _jsonl(out)[0]
  context = qa._contexts(contexts)[0][0]
  qa._begin_pending(
    qa._pending_path(out),
    question_id=first["question_id"],
    request_id=first["request_id"],
    prompt=qa._reader_prompt(context),
  )
  monkeypatch.delenv("OPENAI_API_KEY")
  qa._read(args)

  record = _jsonl(out)[0]
  receipt = json.loads(out.with_suffix(".jsonl.receipt.json").read_text())
  digest = receipt.pop("receipt_sha256")
  assert len(calls) == 1
  assert record["question_id"] == "degree"
  assert record["hypothesis"] == "Business Administration."
  assert record["context_run_id"] == RUN_ID
  assert record["cost"] == {"estimated_usd": 0.0, "reserved_usd": 0.0}
  assert not out.with_suffix(".jsonl.pending.json").exists()
  assert digest == qa._digest(receipt)
  assert receipt["schema"] == "agos-memory-lab-read-v2"
  assert receipt["config"] == {
    "request": {
      "base_url": "https://api.openai.com/v1",
      "model": "reader-v1",
      "temperature": 0.0,
      "max_tokens": 1_000,
    },
    "execution": {
      "timeout": 120.0,
      "concurrency": 1,
      "retries": 0,
      "input_cost_per_million": 0.0,
      "output_cost_per_million": 0.0,
      "max_cost_usd": 0.0,
    },
  }
  assert "secret" not in out.read_text()
  assert "secret" not in json.dumps(receipt)


def test_reader_rejects_changed_context_before_calling_model(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  contexts = tmp_path / "contexts.jsonl"
  out = tmp_path / "hypotheses.jsonl"
  _write_jsonl(contexts, [_context("degree", "What degree?", "Business Administration")])
  args = _args("read", "--contexts", str(contexts), "--out", str(out), "--model", "reader-v1")
  calls = 0

  def chat(*_: object, **__: object) -> qa.ChatResult:
    nonlocal calls
    calls += 1
    return qa.ChatResult("Business Administration.", "reader-v1", 20, 3, 23)

  monkeypatch.setattr(qa, "_chat", chat)
  monkeypatch.setenv("OPENAI_API_KEY", "secret")
  qa._read(args)
  _write_jsonl(contexts, [_context("degree", "What degree?", "Different context")])

  with pytest.raises(qa.QAError, match="^reader_resume_identity_mismatch$"):
    qa._read(args)

  assert calls == 1


def test_reader_rejects_corrupt_checkpoint_before_calling_model(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  contexts = tmp_path / "contexts.jsonl"
  out = tmp_path / "hypotheses.jsonl"
  _write_jsonl(contexts, [_context("degree", "What degree?", "Business Administration")])
  args = _args("read", "--contexts", str(contexts), "--out", str(out), "--model", "reader-v1")
  monkeypatch.setenv("OPENAI_API_KEY", "secret")
  monkeypatch.setattr(
    qa,
    "_chat",
    lambda *_args, **_kwargs: qa.ChatResult("Business Administration.", "reader-v1", 20, 3, 23),
  )
  qa._read(args)
  record = _jsonl(out)[0]
  record["hypothesis"] = "altered"
  _write_jsonl(out, [record])

  with pytest.raises(qa.QAError, match="^hypothesis_resume_invalid$"):
    qa._read(args)


def test_reader_semantic_run_identity_excludes_runtime_measurements(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  contexts = tmp_path / "contexts.jsonl"
  _write_jsonl(contexts, [_context("degree", "What degree?", "Business Administration")])
  monkeypatch.setenv("OPENAI_API_KEY", "secret")
  monkeypatch.setattr(
    qa,
    "_chat",
    lambda *_args, **_kwargs: qa.ChatResult("Business Administration.", "reader-v1", 20, 3, 23),
  )
  run_ids = []
  for name in ("first", "second"):
    out = tmp_path / f"{name}.jsonl"
    args = _args("read", "--contexts", str(contexts), "--out", str(out), "--model", "reader-v1")
    qa._read(args)
    run_ids.append(json.loads(out.with_suffix(".jsonl.receipt.json").read_text())["run_id"])

  assert run_ids[0] == run_ids[1]


def test_request_identity_excludes_execution_policy() -> None:
  context = qa.Context(RUN_ID, "degree", "What degree?", "2024/01/03", "context", "b" * 64)
  reference = qa.Reference("degree", "single-session-user", "What degree?", "Business", False)
  config = _chat_config()
  execution = replace(
    config,
    timeout=30,
    input_cost=2,
    output_cost=4,
    max_cost=20,
  )

  assert qa._reader_id(context, execution) == qa._reader_id(context, config)
  assert qa._judge_id(reference, "Business", execution) == qa._judge_id(reference, "Business", config)


@pytest.mark.parametrize(
  "change",
  [
    {"base_url": "https://example.com/v1"},
    {"model": "reader-v2"},
    {"temperature": 0.5},
    {"max_tokens": 999},
  ],
)
def test_request_identity_includes_request_semantics(change: dict[str, object]) -> None:
  context = qa.Context(RUN_ID, "degree", "What degree?", "2024/01/03", "context", "b" * 64)
  reference = qa.Reference("degree", "single-session-user", "What degree?", "Business", False)
  config = _chat_config()
  changed = replace(config, **change)

  assert qa._reader_id(context, changed) != qa._reader_id(context, config)
  assert qa._judge_id(reference, "Business", changed) != qa._judge_id(reference, "Business", config)


def test_unknown_request_outcome_blocks_an_automatic_repeat(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  contexts = tmp_path / "contexts.jsonl"
  out = tmp_path / "hypotheses.jsonl"
  _write_jsonl(contexts, [_context("degree", "What degree?", "Business Administration")])
  args = _args("read", "--contexts", str(contexts), "--out", str(out), "--model", "reader-v1")
  calls = 0

  def fail(*_: object, **__: object) -> qa.ChatResult:
    nonlocal calls
    calls += 1
    raise qa.QAError("chat_request_failed:TimeoutError")

  monkeypatch.setenv("OPENAI_API_KEY", "secret")
  monkeypatch.setattr(qa, "_chat", fail)
  with pytest.raises(qa.QAError, match="^chat_request_failed:TimeoutError$"):
    qa._read(args)

  monkeypatch.setattr(
    qa,
    "_chat",
    lambda *_args, **_kwargs: qa.ChatResult("Business Administration.", "reader-v1", 20, 3, 23),
  )
  with pytest.raises(qa.QAError, match="^chat_request_outcome_unknown$"):
    qa._read(args)

  assert calls == 1
  assert out.with_suffix(".jsonl.pending.json").exists()


def test_official_judge_scores_and_exact_resume_makes_no_second_call(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  hypotheses = tmp_path / "hypotheses.jsonl"
  out = tmp_path / "evaluation.jsonl"
  _write_jsonl(
    hypotheses,
    [
      {"question_id": "degree", "hypothesis": "Business Administration."},
      {"question_id": "update", "hypothesis": "Green."},
      {"question_id": "missing_abs", "hypothesis": "The history does not say."},
    ],
  )
  args = _args(
    "judge",
    "--hypotheses",
    str(hypotheses),
    "--references",
    str(FIXTURE),
    "--sha256",
    FIXTURE_SHA256,
    "--revision",
    "fixture-v1",
    "--out",
    str(out),
    "--model",
    "judge-v1",
  )
  calls: list[str] = []

  def chat(prompt: str, **_: object) -> qa.ChatResult:
    calls.append(prompt)
    return qa.ChatResult("yes", "judge-v1", 30, 1, 31)

  monkeypatch.setattr(qa, "_chat", chat)
  monkeypatch.setenv("OPENAI_API_KEY", "secret")
  qa._judge(args)
  monkeypatch.delenv("OPENAI_API_KEY")
  qa._judge(args)

  receipt = json.loads(out.with_suffix(".jsonl.receipt.json").read_text())
  summary = receipt["summary"]
  assert len(calls) == 3
  assert summary["accuracy"] == 1.0
  assert summary["task_accuracy"] == 1.0
  assert summary["abstention_accuracy"] == 1.0
  assert summary["by_type"]["single-session-user"] == {"cases": 2, "accuracy": 1.0}
  assert summary["by_type"]["knowledge-update"] == {"cases": 1, "accuracy": 1.0}
  assert summary["strict_parse_failures"] == 0
  assert summary["strict_parse_disagreements"] == 0
  assert "previous information" in calls[1]
  assert "unanswerable question" in calls[2]


@pytest.mark.parametrize(
  "value,error",
  [
    ({}, "chat_response_invalid"),
    ({"choices": []}, "chat_response_invalid"),
    ({"choices": [{"message": {"content": ""}}], "model": "m"}, "chat_response_invalid"),
    (
      {
        "choices": [{"message": {"content": "answer"}}],
        "model": "m",
        "usage": {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 1},
      },
      "chat_usage_invalid",
    ),
  ],
)
def test_chat_response_validation_is_fail_closed(value: object, error: str) -> None:
  with pytest.raises(qa.QAError, match=f"^{error}$"):
    qa._chat_result(value)


def test_remote_plaintext_endpoint_is_rejected() -> None:
  args = _args(
    "read",
    "--contexts",
    "contexts.jsonl",
    "--out",
    "out.jsonl",
    "--model",
    "reader-v1",
    "--base-url",
    "http://example.com/v1",
  )

  with pytest.raises(qa.QAError, match="^chat_base_url_invalid$"):
    qa._config(args)


def test_input_cannot_be_overwritten_by_output(tmp_path: Path) -> None:
  contexts = tmp_path / "contexts.jsonl"
  _write_jsonl(contexts, [_context("degree", "What degree?", "Business Administration")])
  args = _args("read", "--contexts", str(contexts), "--out", str(contexts), "--model", "reader-v1")

  with pytest.raises(qa.QAError, match="^input_output_path_conflict$"):
    qa._read(args)


def test_hard_cost_cap_is_checked_before_a_call() -> None:
  config = replace(_chat_config(), input_cost=1, output_cost=1, max_cost=0.001)

  with pytest.raises(qa.QAError, match="^chat_cost_cap_reached$"):
    qa._check_cost((), prompt="question", config=config)


def test_cost_reservation_is_one_explicit_request_bound() -> None:
  config = replace(_chat_config(), max_tokens=100, input_cost=2, output_cost=3, max_cost=1)

  assert qa._reserved_cost("é", config=config) == 0.000816


def test_completed_actual_cost_releases_unused_reservation() -> None:
  config = replace(_chat_config(), input_cost=1, output_cost=1, max_cost=0.002)
  completed = ({"cost": {"estimated_usd": 0.0001, "reserved_usd": 0.9}},)

  qa._check_cost(completed, prompt="question", config=config)

  unknown = ({"cost": {"estimated_usd": None, "reserved_usd": 0.9}},)
  with pytest.raises(qa.QAError, match="^chat_cost_cap_reached$"):
    qa._check_cost(unknown, prompt="question", config=config)


def test_official_label_is_reported_separately_from_strict_parse() -> None:
  assert qa._labels("not yes") == (True, None)
  assert qa._labels("No. The answer is unsupported.") == (False, False)


def _args(*values: str):
  return qa._parser().parse_args(values)


def _chat_config() -> qa.ChatConfig:
  return qa.ChatConfig(
    base_url="https://api.openai.com/v1",
    model="reader-v1",
    temperature=0,
    max_tokens=1_000,
    timeout=120,
    input_cost=0,
    output_cost=0,
    max_cost=0,
  )


def _context(question_id: str, question: str, context: str) -> dict[str, str]:
  return {
    "run_id": RUN_ID,
    "question_id": question_id,
    "question": question,
    "question_date": "2024/01/03 (Wed) 12:00",
    "context": context,
    "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
  }


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
  path.write_text("".join(f"{json.dumps(value)}\n" for value in values))


def _jsonl(path: Path) -> list[dict[str, object]]:
  return [json.loads(line) for line in path.read_text().splitlines()]
