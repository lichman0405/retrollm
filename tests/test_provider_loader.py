from __future__ import annotations

import pytest

from retrollm.llm.provider_loader import LLMSettings, build_provider
from retrollm.llm.providers.openai_compatible import OpenAICompatibleProvider


def _settings(provider: str, base_url: str = "https://api.example.com", api_key: str = "k") -> LLMSettings:
    return LLMSettings(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model="test-model",
        temperature=0.2,
        timeout_s=60,
    )


def test_build_provider_accepts_deepseek_alias() -> None:
    provider = build_provider(_settings("DeepSeek"))
    assert isinstance(provider, OpenAICompatibleProvider)


def test_build_provider_plain_provider_falls_back_to_openai_compatible_with_base_url() -> None:
    provider = build_provider(_settings("my_inhouse_gateway"))
    assert isinstance(provider, OpenAICompatibleProvider)


def test_build_provider_unknown_plain_provider_without_base_url_errors() -> None:
    with pytest.raises(ValueError, match="Unknown provider alias"):
        build_provider(_settings("unknown_provider", base_url="", api_key="k"))
