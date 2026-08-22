from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

import call as bounded
import model


_SCHEMA = "agos-memory-lab-endpoint-qualification-v1"
_PENDING_SCHEMA = "agos-memory-lab-endpoint-qualification-pending-v1"
_NONCE = "agos-memory-lab-contract-v1"
_PROMPT = (
  "This is an API contract check. Return the required structured output with nonce exactly "
  f"{_NONCE}. Do not add other fields."
)
_ENVELOPE_BYTES = 256


class Reply(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

  nonce: Literal["agos-memory-lab-contract-v1"]


class QualificationError(Exception):
  pass


def main() -> None:
  parser = _parser()
  args = parser.parse_args()
  try:
    value = _plan(args) if args.plan else _qualify(args)
    print(json.dumps(value, indent=2, sort_keys=True))
  except (QualificationError, bounded.CallError, model.ModelError) as exc:
    raise SystemExit(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Qualify one OpenAI-compatible endpoint contract.")
  parser.add_argument("--plan", action="store_true", help="Print the request contract without credentials or a call.")
  parser.add_argument("--out", type=Path, help="Immutable qualification receipt; required unless --plan.")
  bounded.arguments(parser, default_tokens=64)
  return parser


def _plan(args: argparse.Namespace) -> dict[str, Any]:
  config = bounded.config(args)
  reserved = bounded.reserve(_PROMPT, config=config, overhead=_overhead())
  priced = config.input_cost > 0 or config.output_cost > 0
  return {
    "schema": "agos-memory-lab-endpoint-qualification-plan-v1",
    "config": {
      "request": bounded.request(config),
      "execution": bounded.execution(config),
    },
    "prompt_sha256": _text_digest(_PROMPT),
    "output_schema_sha256": _digest(Reply.model_json_schema()),
    "calls": 1,
    "reserved_cost_usd": reserved,
    "fits_cost_cap": not priced or reserved <= config.max_cost,
  }


def _qualify(args: argparse.Namespace) -> dict[str, Any]:
  if args.out is None:
    raise QualificationError("qualification_output_required")
  if args.out.exists():
    raise QualificationError("qualification_output_exists")
  pending = _pending_path(args.out)
  if pending.exists():
    raise QualificationError("qualification_request_outcome_unknown")
  config = bounded.config(args)
  bounded.check((), prompt=_PROMPT, config=config, overhead=_overhead())
  key = bounded.key(args, config=config)
  request_id = _digest(
    {
      "schema": _SCHEMA,
      "request": bounded.request(config),
      "prompt_sha256": _text_digest(_PROMPT),
      "output_schema_sha256": _digest(Reply.model_json_schema()),
    }
  )
  bounded.begin(
    pending,
    {
      "schema": _PENDING_SCHEMA,
      "request_id": request_id,
      "prompt_sha256": _text_digest(_PROMPT),
    },
    unknown="qualification_request_outcome_unknown",
    failed="qualification_pending_write_failed",
  )
  started = time.perf_counter()
  result = model.structure(_PROMPT, output_type=Reply, config=config, api_key=key)
  latency = round(time.perf_counter() - started, 6)
  semantic = {
    "schema": _SCHEMA,
    "request_id": request_id,
    "config": {
      "request": bounded.request(config),
      "execution": bounded.execution(config),
    },
    "prompt_sha256": _text_digest(_PROMPT),
    "output_schema_sha256": _digest(Reply.model_json_schema()),
    "result": {
      "response_model": result.model,
      "nonce_sha256": _text_digest(result.content.nonce),
      "usage": bounded.usage(result),
      "cost": bounded.cost(result, prompt=_PROMPT, config=config, overhead=_overhead()),
    },
  }
  receipt = {
    **semantic,
    "run_id": _digest(semantic),
    "measurements": {"latency_seconds": latency},
  }
  receipt = {**receipt, "receipt_sha256": _digest(receipt)}
  _write(args.out, receipt)
  bounded.clear(pending, error="qualification_pending_clear_failed")
  return receipt


def _overhead() -> int:
  schema = json.dumps(Reply.model_json_schema(), separators=(",", ":"), sort_keys=True)
  return len(schema.encode("utf-8")) + _ENVELOPE_BYTES


def _pending_path(path: Path) -> Path:
  return path.with_suffix(f"{path.suffix}.pending.json")


def _write(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix(f"{path.suffix}.part")
  partial.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
  os.replace(partial, path)


def _text_digest(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
  encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
  return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


if __name__ == "__main__":
  main()
