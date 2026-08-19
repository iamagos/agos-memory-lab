import json
from collections.abc import Callable

import httpx
import pytest

import model


def test_openai_completion_uses_one_bounded_chat_request(monkeypatch: pytest.MonkeyPatch) -> None:
  requests: list[httpx.Request] = []

  def respond(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    return httpx.Response(200, json=_response(), request=request)

  _transport(monkeypatch, respond)
  result = model.complete("hello", config=_config(), api_key="secret")

  assert result == model.ModelResult("answer", "served-model", 13, 2, 15)
  assert len(requests) == 1
  assert str(requests[0].url) == "https://api.openai.com/v1/chat/completions"
  assert requests[0].headers["authorization"] == "Bearer secret"
  assert json.loads(requests[0].content) == {
    "messages": [{"role": "user", "content": "hello"}],
    "model": "reader-deployment",
    "max_completion_tokens": 20,
    "stream": False,
  }


def test_explicit_supported_temperature_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
  requests: list[httpx.Request] = []

  def respond(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    return httpx.Response(200, json=_response(), request=request)

  _transport(monkeypatch, respond)

  model.complete("hello", config=_config(temperature=0.0), api_key="secret")

  assert json.loads(requests[0].content)["temperature"] == 0.0


@pytest.mark.parametrize(
  "change",
  (
    {},
    {"provider": "azure", "base_url": "https://resource.openai.azure.com/openai/v1"},
  ),
)
def test_reasoning_model_rejects_temperature_before_request(
  monkeypatch: pytest.MonkeyPatch, change: dict[str, object]
) -> None:
  requests: list[httpx.Request] = []

  def respond(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    return httpx.Response(200, json=_response(), request=request)

  _transport(monkeypatch, respond)

  with pytest.raises(model.ModelError, match="^chat_temperature_unsupported$"):
    model.complete(
      "hello",
      config=_config(model="gpt-5-mini", temperature=0.0, **change),
      api_key="secret",
    )

  assert requests == []


@pytest.mark.parametrize(
  "base_url,expected",
  (
    (
      "https://resource.openai.azure.com/openai/v1/",
      "https://resource.openai.azure.com/openai/v1/chat/completions",
    ),
    (
      "https://deployment.eastus.models.ai.azure.com",
      "https://deployment.eastus.models.ai.azure.com/v1/chat/completions",
    ),
  ),
)
def test_azure_v1_uses_openai_compatible_endpoint(
  monkeypatch: pytest.MonkeyPatch, base_url: str, expected: str
) -> None:
  requests: list[httpx.Request] = []

  def respond(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    return httpx.Response(200, json=_response(), request=request)

  _transport(monkeypatch, respond)
  config = _config(provider="azure", base_url=base_url)

  model.complete("hello", config=config, api_key="secret")

  assert len(requests) == 1
  assert str(requests[0].url) == expected
  assert requests[0].headers["authorization"] == "Bearer secret"
  assert "api-version" not in requests[0].url.params


def test_azure_legacy_uses_deployment_and_api_version(monkeypatch: pytest.MonkeyPatch) -> None:
  requests: list[httpx.Request] = []

  def respond(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    return httpx.Response(200, json=_response(), request=request)

  _transport(monkeypatch, respond)
  config = _config(
    provider="azure",
    base_url="https://resource.openai.azure.com/",
    api_version="2025-04-01-preview",
  )

  model.complete("hello", config=config, api_key="secret")

  assert len(requests) == 1
  assert requests[0].url.path == "/openai/deployments/reader-deployment/chat/completions"
  assert dict(requests[0].url.params) == {"api-version": "2025-04-01-preview"}
  assert requests[0].headers["api-key"] == "secret"


@pytest.mark.parametrize(
  "change",
  (
    {},
    {"provider": "azure", "base_url": "https://resource.openai.azure.com/openai/v1"},
    {
      "provider": "azure",
      "base_url": "https://resource.openai.azure.com",
      "api_version": "2025-04-01-preview",
    },
  ),
)
def test_provider_errors_are_not_retried(
  monkeypatch: pytest.MonkeyPatch, change: dict[str, object]
) -> None:
  calls = 0

  def fail(request: httpx.Request) -> httpx.Response:
    nonlocal calls
    calls += 1
    return httpx.Response(500, json={"error": {"message": "failed", "type": "server_error"}}, request=request)

  _transport(monkeypatch, fail)

  with pytest.raises(model.ModelError, match="^chat_http_error:500$"):
    model.complete("hello", config=_config(**change), api_key="secret")

  assert calls == 1


def test_redirect_is_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
  calls = 0

  def redirect(request: httpx.Request) -> httpx.Response:
    nonlocal calls
    calls += 1
    return httpx.Response(307, headers={"location": "https://example.com/elsewhere"}, request=request)

  _transport(monkeypatch, redirect)

  with pytest.raises(model.ModelError, match="^chat_request_failed:ModelAPIError$"):
    model.complete("hello", config=_config(), api_key="secret")

  assert calls == 1


def test_missing_provider_usage_remains_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
  def respond(request: httpx.Request) -> httpx.Response:
    payload = _response()
    payload.pop("usage")
    return httpx.Response(200, json=payload, request=request)

  _transport(monkeypatch, respond)

  result = model.complete("hello", config=_config(), api_key="secret")

  assert (result.input_tokens, result.output_tokens, result.total_tokens) == (None, None, None)


@pytest.mark.parametrize(
  "change,error",
  [
    ({"provider": "azure", "base_url": "https://resource.openai.azure.com"}, "chat_api_version_required"),
    (
      {
        "provider": "azure",
        "base_url": "https://resource.openai.azure.com/openai/v1",
        "api_version": "2025-04-01-preview",
      },
      "chat_api_version_unexpected",
    ),
    ({"api_version": "2025-04-01-preview"}, "chat_api_version_unexpected"),
    (
      {"provider": "azure", "base_url": "http://127.0.0.1:8000", "api_version": "2025-04-01-preview"},
      "chat_base_url_invalid",
    ),
    ({"provider": []}, "chat_provider_invalid"),
    ({"base_url": 1}, "chat_base_url_invalid"),
    ({"api_version": 1}, "chat_api_version_invalid"),
    ({"temperature": True}, "chat_temperature_invalid"),
    ({"timeout": "30"}, "chat_timeout_invalid"),
  ],
)
def test_endpoint_families_fail_closed(change: dict[str, object], error: str) -> None:
  values = {
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "api_version": None,
    "model": "reader-deployment",
    "temperature": None,
    "max_tokens": 20,
    "timeout": 30.0,
    **change,
  }

  with pytest.raises(model.ModelError, match=f"^{error}$"):
    model.ModelConfig(**values)  # type: ignore[arg-type]


def _transport(
  monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
  transport = httpx.MockTransport(handler)
  monkeypatch.setattr(model, "_http", lambda _config: httpx.AsyncClient(transport=transport))


def _config(**change: object) -> model.ModelConfig:
  values = {
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "api_version": None,
    "model": "reader-deployment",
    "temperature": None,
    "max_tokens": 20,
    "timeout": 30.0,
    **change,
  }
  return model.ModelConfig(**values)  # type: ignore[arg-type]


def _response() -> dict[str, object]:
  return {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "served-model",
    "choices": [
      {
        "index": 0,
        "message": {"role": "assistant", "content": "answer"},
        "finish_reason": "stop",
      }
    ],
    "usage": {"prompt_tokens": 13, "completion_tokens": 2, "total_tokens": 15},
  }
