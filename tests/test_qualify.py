import json
from pathlib import Path

import pytest

import model
import qualify


def test_plan_freezes_custom_endpoint_without_credentials_or_writes(tmp_path: Path) -> None:
  args = _args(
    "--plan",
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


def _args(*values: str):
  return qualify._parser().parse_args(values)
