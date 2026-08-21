import argparse

import pytest

import call


def test_official_services_receive_stable_default_identities() -> None:
  assert call.config(_args("--model", "gpt-5")).provider_id == "openai"
  assert call.config(_args("--provider", "deepseek", "--model", "deepseek-v4-flash")).provider_id == "deepseek"
  assert call.config(
    _args(
      "--provider",
      "azure",
      "--base-url",
      "https://resource.openai.azure.com/openai/v1",
      "--model",
      "deployment",
    )
  ).provider_id == "azure-openai"


def test_deepseek_provider_uses_isolated_default_key(monkeypatch: pytest.MonkeyPatch) -> None:
  args = _args("--provider", "deepseek", "--model", "deepseek-v4-flash")
  monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

  assert call.key(args, config=call.config(args)) == "secret"


def test_custom_compatible_endpoint_requires_explicit_service_identity() -> None:
  args = _args(
    "--base-url",
    "https://api.deepseek.com/v1",
    "--model",
    "deepseek-chat",
  )

  with pytest.raises(call.CallError, match="^chat_provider_id_required$"):
    call.config(args)


def test_custom_service_identity_is_bound_to_request_and_key_name(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  args = _args(
    "--provider-id",
    "deepseek",
    "--base-url",
    "https://api.deepseek.com/v1",
    "--model",
    "deepseek-chat",
  )
  config = call.config(args)

  assert call.request(config)["provider_id"] == "deepseek"
  with pytest.raises(call.CallError, match="^chat_api_key_env_required$"):
    call.key(args, config=config)

  named = _args(
    "--provider-id",
    "deepseek",
    "--base-url",
    "https://api.deepseek.com/v1",
    "--api-key-env",
    "DEEPSEEK_API_KEY",
    "--model",
    "deepseek-chat",
  )
  monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
  assert call.key(named, config=call.config(named)) == "secret"


@pytest.mark.parametrize("provider_id", ("DeepSeek", "bad id", "", "-bad"))
def test_service_identity_is_a_lowercase_receipt_slug(provider_id: str) -> None:
  args = _args(f"--provider-id={provider_id}", "--model", "model")

  with pytest.raises(call.CallError, match="^chat_provider_id_invalid$"):
    call.config(args)


def _args(*values: str) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  call.arguments(parser, default_tokens=100)
  return parser.parse_args(values)
