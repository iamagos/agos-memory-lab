import json
from pathlib import Path

import pytest

import model
import qualify


def test_plan_freezes_custom_endpoint_without_credentials_or_writes(tmp_path: Path) -> None:
  args = _args(
    "--plan",
    "--provider",
    "deepseek",
    "--provider-id",
    "deepseek",
    "--base-url",
    "https://api.deepseek.com/v1",
    "--model",
    "deepseek-chat",
    "--input-cost",
    "1",
    "--output-cost",
    "2",
    "--max-cost",
    "1",
  )

  plan = qualify._plan(args)

  assert plan["calls"] == 1
  assert plan["config"]["request"]["provider"] == "deepseek"
  assert plan["config"]["request"]["provider_id"] == "deepseek"
  assert plan["config"]["request"]["model"] == "deepseek-chat"
  assert plan["reserved_cost_usd"] > 0
  assert not tuple(tmp_path.iterdir())


def test_qualification_makes_one_strict_call_and_records_served_model(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  out = tmp_path / "qualification.json"
  args = _args("--out", str(out), "--model", "requested-model")
  calls = 0

  def chat(*_: object, **__: object) -> model.ModelResult[qualify.Reply]:
    nonlocal calls
    calls += 1
    return model.ModelResult(qualify.Reply(nonce=qualify._NONCE), "served-model", 20, 2, 22)

  monkeypatch.setenv("OPENAI_API_KEY", "secret")
  monkeypatch.setattr(qualify.model, "structure", chat)

  receipt = qualify._qualify(args)

  assert calls == 1
  assert receipt["result"]["response_model"] == "served-model"
  assert receipt["result"]["usage"] == {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22}
  assert json.loads(out.read_text(encoding="utf-8")) == receipt
  assert not qualify._pending_path(out).exists()


def test_unknown_qualification_outcome_blocks_another_call(tmp_path: Path) -> None:
  out = tmp_path / "qualification.json"
  pending = qualify._pending_path(out)
  pending.write_bytes(b"{}\n")
  args = _args("--out", str(out), "--model", "model")

  with pytest.raises(qualify.QualificationError, match="^qualification_request_outcome_unknown$"):
    qualify._qualify(args)


def test_reconciliation_closes_one_matching_pending_request_without_a_call(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  out = tmp_path / "qualification.json"
  args = _args(
    "--out",
    str(out),
    "--model",
    "model",
    "--reconcile-error",
    "chat_output_limit_exceeded",
  )
  config = qualify.bounded.config(args)
  pending = qualify._pending_path(out)
  pending.write_text(
    json.dumps(
      {
        "schema": qualify._PENDING_SCHEMA,
        "request_id": qualify._request_id(config),
        "prompt_sha256": qualify._text_digest(qualify._PROMPT),
      },
      sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
  )

  def unexpected(*_: object, **__: object) -> object:
    raise AssertionError("reconciliation_made_a_call")

  monkeypatch.setattr(qualify.model, "structure", unexpected)
  receipt = qualify._reconcile(args)

  assert receipt["schema"] == qualify._FAILURE_SCHEMA
  assert receipt["result"] == {
    "status": "failed",
    "error": "chat_output_limit_exceeded",
    "usage": None,
    "cost": {"estimated_usd": None, "reserved_usd": 0.0},
  }
  assert json.loads(out.read_text(encoding="utf-8")) == receipt
  assert not pending.exists()


def test_reconciliation_preserves_a_mismatched_pending_request(tmp_path: Path) -> None:
  out = tmp_path / "qualification.json"
  pending = qualify._pending_path(out)
  pending.write_text(
    json.dumps(
      {
        "schema": qualify._PENDING_SCHEMA,
        "request_id": "different-request",
        "prompt_sha256": qualify._text_digest(qualify._PROMPT),
      }
    )
    + "\n",
    encoding="utf-8",
  )
  args = _args(
    "--out",
    str(out),
    "--model",
    "model",
    "--reconcile-error",
    "chat_http_400",
  )

  with pytest.raises(qualify.QualificationError, match="^qualification_pending_request_mismatch$"):
    qualify._reconcile(args)

  assert pending.exists()
  assert not out.exists()


def _args(*values: str):
  return qualify._parser().parse_args(values)
