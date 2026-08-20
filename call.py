from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import model


@dataclass(frozen=True, slots=True)
class Config(model.ModelConfig):
  input_cost: float
  output_cost: float
  max_cost: float

  def __post_init__(self) -> None:
    model.ModelConfig.__post_init__(self)
    for value in (self.input_cost, self.output_cost, self.max_cost):
      if not _number(value) or not math.isfinite(value) or value < 0:
        raise CallError("chat_cost_invalid")
    if (self.input_cost > 0 or self.output_cost > 0) and self.max_cost <= 0:
      raise CallError("chat_cost_cap_required")


class CallError(Exception):
  pass


def arguments(parser: argparse.ArgumentParser, *, default_tokens: int) -> None:
  parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
  parser.add_argument("--model", required=True)
  parser.add_argument("--base-url")
  parser.add_argument("--api-version")
  parser.add_argument("--api-key-env")
  parser.add_argument("--temperature", type=float)
  parser.add_argument("--max-tokens", type=int, default=default_tokens)
  parser.add_argument("--timeout", type=float, default=120.0)
  parser.add_argument("--input-cost", type=float, default=0.0, help="USD per million input tokens.")
  parser.add_argument("--output-cost", type=float, default=0.0, help="USD per million output tokens.")
  parser.add_argument("--max-cost", type=float, default=0.0, help="Declared maximum estimated USD.")


def config(args: argparse.Namespace) -> Config:
  base_url = args.base_url or ("https://api.openai.com/v1" if args.provider == "openai" else None)
  if base_url is None:
    raise CallError("chat_base_url_required")
  try:
    return Config(
      provider=args.provider,
      base_url=base_url,
      api_version=args.api_version,
      model=args.model,
      temperature=args.temperature,
      max_tokens=args.max_tokens,
      timeout=args.timeout,
      input_cost=args.input_cost,
      output_cost=args.output_cost,
      max_cost=args.max_cost,
    )
  except model.ModelError as exc:
    raise CallError(str(exc)) from exc


def key(args: argparse.Namespace, *, config: Config) -> str:
  name = args.api_key_env or (
    "AZURE_OPENAI_API_KEY" if config.provider == "azure" else "OPENAI_API_KEY"
  )
  if not isinstance(name, str) or not name.strip():
    raise CallError("chat_api_key_env_invalid")
  value = os.getenv(name)
  if not value:
    raise CallError(f"chat_api_key_missing:{name}")
  if any(ord(character) < 32 or ord(character) == 127 for character in value):
    raise CallError("chat_api_key_invalid")
  return value


def request(config: Config) -> dict[str, Any]:
  return {
    "adapter": {
      "api": model.API,
      "openai": model.OPENAI_VERSION,
      "pydantic_ai": model.PYDANTIC_AI_VERSION,
    },
    "provider": config.provider,
    "base_url": config.base_url,
    "api_version": config.api_version,
    "model": config.model,
    "temperature": config.temperature,
    "max_tokens": config.max_tokens,
  }


def execution(config: Config) -> dict[str, Any]:
  return {
    "timeout": config.timeout,
    "concurrency": 1,
    "retries": 0,
    "input_cost_per_million": config.input_cost,
    "output_cost_per_million": config.output_cost,
    "max_cost_usd": config.max_cost,
  }


def usage(result: model.ModelResult[Any]) -> dict[str, int | None]:
  return {
    "input_tokens": result.input_tokens,
    "output_tokens": result.output_tokens,
    "total_tokens": result.total_tokens,
  }


def check(records: Any, *, prompt: str, config: Config, overhead: int) -> None:
  if config.input_cost == 0 and config.output_cost == 0:
    return
  spent = sum(
    record["cost"]["estimated_usd"]
    if record["cost"]["estimated_usd"] is not None
    else record["cost"]["reserved_usd"]
    for record in records
  )
  if spent + reserve(prompt, config=config, overhead=overhead) > config.max_cost:
    raise CallError("chat_cost_cap_reached")


def cost(
  result: model.ModelResult[Any],
  *,
  prompt: str,
  config: Config,
  overhead: int,
) -> dict[str, float | None]:
  estimated = estimate(result, config=config)
  reserved = reserve(prompt, config=config, overhead=overhead)
  if estimated is not None and estimated > reserved:
    raise CallError("chat_cost_bound_exceeded")
  return {
    "estimated_usd": round(estimated, 12) if estimated is not None else None,
    "reserved_usd": reserved,
  }


def estimate(result: model.ModelResult[Any], *, config: Config) -> float | None:
  if (config.input_cost > 0 and result.input_tokens is None) or (
    config.output_cost > 0 and result.output_tokens is None
  ):
    return None
  return (
    (result.input_tokens or 0) * config.input_cost
    + (result.output_tokens or 0) * config.output_cost
  ) / 1_000_000


def reserve(prompt: str, *, config: Config, overhead: int) -> float:
  if not isinstance(overhead, int) or isinstance(overhead, bool) or overhead < 0:
    raise CallError("chat_cost_overhead_invalid")
  input_tokens = len(prompt.encode()) + overhead
  value = (input_tokens * config.input_cost + config.max_tokens * config.output_cost) / 1_000_000
  return math.ceil(value * 1_000_000_000_000) / 1_000_000_000_000


def validate(record: dict[str, Any], *, prompt: str, config: Config, overhead: int, error: str) -> None:
  validate_usage(record.get("usage"), error=error)
  validate_cost(record.get("cost"), error=error)
  usage_value = record["usage"]
  result = model.ModelResult(
    None,
    "validated",
    usage_value["input_tokens"],
    usage_value["output_tokens"],
    usage_value["total_tokens"],
  )
  try:
    expected = cost(result, prompt=prompt, config=config, overhead=overhead)
  except CallError as exc:
    raise CallError(error) from exc
  if record["cost"] != expected:
    raise CallError(error)


def validate_usage(value: Any, *, error: str) -> None:
  if not isinstance(value, dict) or set(value) != {"input_tokens", "output_tokens", "total_tokens"}:
    raise CallError(error)
  counts = tuple(_optional_count(value[name], error=error) for name in ("input_tokens", "output_tokens", "total_tokens"))
  if counts[2] is not None and any(count is not None and count > counts[2] for count in counts[:2]):
    raise CallError(error)


def validate_cost(value: Any, *, error: str) -> None:
  if not isinstance(value, dict) or set(value) != {"estimated_usd", "reserved_usd"}:
    raise CallError(error)
  estimated = value["estimated_usd"]
  reserved = value["reserved_usd"]
  if estimated is not None and (not _number(estimated) or not math.isfinite(estimated) or estimated < 0):
    raise CallError(error)
  if not _number(reserved) or not math.isfinite(reserved) or reserved < 0:
    raise CallError(error)
  if estimated is not None and estimated > reserved:
    raise CallError(error)


def validate_duration(value: Any, *, error: str) -> None:
  if not _number(value) or not math.isfinite(value) or value < 0:
    raise CallError(error)


def begin(path: Path, value: dict[str, Any], *, unknown: str, failed: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  try:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as target:
      target.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
      target.flush()
      os.fsync(target.fileno())
  except FileExistsError as exc:
    raise CallError(unknown) from exc
  except OSError as exc:
    raise CallError(failed) from exc


def pending(path: Path, *, error: str) -> dict[str, Any] | None:
  if not path.exists():
    return None
  try:
    value = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError) as exc:
    raise CallError(error) from exc
  if not isinstance(value, dict):
    raise CallError(error)
  return value


def clear(path: Path, *, error: str) -> None:
  try:
    path.unlink()
  except OSError as exc:
    raise CallError(error) from exc


def _optional_count(value: Any, *, error: str) -> int | None:
  if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
    raise CallError(error)
  return value


def _number(value: Any) -> bool:
  return isinstance(value, int | float) and not isinstance(value, bool)
