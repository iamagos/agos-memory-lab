from __future__ import annotations

import asyncio
import math
import urllib.parse
from dataclasses import dataclass
from importlib.metadata import version
from typing import Literal

import httpx
from openai import APIConnectionError, APIResponseValidationError, APIStatusError, AsyncOpenAI, OpenAIError
from pydantic_ai import Agent, UnexpectedModelBehavior, UsageLimits
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider


Provider = Literal["openai", "azure"]
API = "chat-completions"
OPENAI_VERSION = version("openai")
PYDANTIC_AI_VERSION = version("pydantic-ai-slim")


class ModelError(Exception):
  pass


@dataclass(frozen=True, slots=True)
class ModelConfig:
  provider: Provider
  base_url: str
  api_version: str | None
  model: str
  temperature: float | None
  max_tokens: int
  timeout: float

  def __post_init__(self) -> None:
    if not isinstance(self.provider, str) or self.provider not in {"openai", "azure"}:
      raise ModelError("chat_provider_invalid")
    if (
      not isinstance(self.base_url, str)
      or not self.base_url
      or self.base_url != self.base_url.strip()
      or _unsafe(self.base_url)
    ):
      raise ModelError("chat_base_url_invalid")
    base_url = self.base_url.rstrip("/")
    parts = urllib.parse.urlsplit(base_url)
    if (
      parts.scheme not in {"http", "https"}
      or not parts.hostname
      or parts.username is not None
      or parts.password is not None
      or parts.query
      or parts.fragment
      or (parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1"})
      or (self.provider == "azure" and parts.scheme != "https")
    ):
      raise ModelError("chat_base_url_invalid")
    if self.api_version is not None and not isinstance(self.api_version, str):
      raise ModelError("chat_api_version_invalid")
    api_version = self.api_version.strip() if isinstance(self.api_version, str) else None
    if api_version == "" or (api_version is not None and _unsafe(api_version)):
      raise ModelError("chat_api_version_invalid")
    if self.provider == "openai" and api_version is not None:
      raise ModelError("chat_api_version_unexpected")
    if self.provider == "azure":
      if _v1_url(base_url) is not None and api_version is not None:
        raise ModelError("chat_api_version_unexpected")
      if _v1_url(base_url) is None and api_version is None:
        raise ModelError("chat_api_version_required")
    if not isinstance(self.model, str) or not self.model.strip() or _unsafe(self.model):
      raise ModelError("chat_model_invalid")
    if self.temperature is not None and (
      not _number(self.temperature) or not math.isfinite(self.temperature) or self.temperature < 0
    ):
      raise ModelError("chat_temperature_invalid")
    if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) or self.max_tokens < 1:
      raise ModelError("chat_token_limit_invalid")
    if not _number(self.timeout) or not math.isfinite(self.timeout) or self.timeout <= 0:
      raise ModelError("chat_timeout_invalid")
    object.__setattr__(self, "base_url", base_url)
    object.__setattr__(self, "api_version", api_version)
    object.__setattr__(self, "model", self.model.strip())


@dataclass(frozen=True, slots=True)
class ModelResult:
  content: str
  model: str
  input_tokens: int | None
  output_tokens: int | None
  total_tokens: int | None


def complete(prompt: str, *, config: ModelConfig, api_key: str) -> ModelResult:
  try:
    return asyncio.run(_complete(prompt, config=config, api_key=api_key))
  except ModelHTTPError as exc:
    raise ModelError(f"chat_http_error:{exc.status_code}") from exc
  except ModelAPIError as exc:
    raise ModelError(f"chat_request_failed:{type(exc).__name__}") from exc
  except APIStatusError as exc:
    raise ModelError(f"chat_http_error:{exc.status_code}") from exc
  except APIConnectionError as exc:
    raise ModelError(f"chat_request_failed:{type(exc).__name__}") from exc
  except UnexpectedModelBehavior as exc:
    raise ModelError("chat_response_invalid") from exc
  except APIResponseValidationError as exc:
    raise ModelError("chat_response_invalid") from exc
  except OpenAIError as exc:
    raise ModelError(f"chat_request_failed:{type(exc).__name__}") from exc


async def _complete(prompt: str, *, config: ModelConfig, api_key: str) -> ModelResult:
  async with _http(config) as http_client:
    provider = _provider(config, api_key=api_key, http_client=http_client)
    chat = OpenAIChatModel(config.model, provider=provider)
    agent = Agent(chat, retries=0)
    result = await agent.run(
      prompt,
      model_settings=_settings(config, chat=chat),
      retries=0,
      usage_limits=UsageLimits(request_limit=1),
    )
  if not isinstance(result.output, str):
    raise ModelError("chat_response_invalid")
  content = result.output.strip()
  response_model = result.response.model_name or config.model
  if not content or not isinstance(response_model, str) or not response_model.strip():
    raise ModelError("chat_response_invalid")
  usage = result.response.usage
  tokens = (usage.input_tokens, usage.output_tokens, usage.total_tokens)
  if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in tokens):
    raise ModelError("chat_usage_invalid")
  if usage.total_tokens == 0:
    tokens = (None, None, None)
  return ModelResult(content, response_model.strip(), *tokens)


def _settings(config: ModelConfig, *, chat: OpenAIChatModel) -> dict[str, float | int]:
  settings: dict[str, float | int] = {"max_tokens": config.max_tokens}
  if config.temperature is None:
    return settings
  profile = chat.profile
  if profile.get("openai_supports_reasoning", False) and profile.get(
    "openai_reasoning_enabled_by_default", False
  ):
    raise ModelError("chat_temperature_unsupported")
  settings["temperature"] = config.temperature
  return settings


def _http(config: ModelConfig) -> httpx.AsyncClient:
  return httpx.AsyncClient(timeout=config.timeout, follow_redirects=False)


def _provider(
  config: ModelConfig, *, api_key: str, http_client: httpx.AsyncClient
) -> OpenAIProvider | AzureProvider:
  if config.provider == "openai":
    client = AsyncOpenAI(
      base_url=config.base_url,
      api_key=api_key,
      http_client=http_client,
      max_retries=0,
    )
    return OpenAIProvider(openai_client=client)
  provider = AzureProvider(
    azure_endpoint=config.base_url,
    api_version=config.api_version,
    api_key=api_key,
    http_client=http_client,
  )
  provider.client.max_retries = 0
  return provider


def _v1_url(base_url: str) -> str | None:
  value = base_url.rstrip("/")
  if value.endswith("/v1"):
    return value
  host = urllib.parse.urlsplit(value).hostname or ""
  if host.endswith(".models.ai.azure.com"):
    return f"{value}/v1"
  return None


def _unsafe(value: str) -> bool:
  return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _number(value: object) -> bool:
  return isinstance(value, int | float) and not isinstance(value, bool)
