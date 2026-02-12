"""Provider loader.

Supports built-in aliases and custom provider classes.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from retrollm.llm.providers.base import LLMProvider
from retrollm.llm.providers.openai_compatible import OpenAICompatibleProvider


_OPENAI_COMPATIBLE_SPEC = (
    "retrollm.llm.providers.openai_compatible:OpenAICompatibleProvider"
)
_ALIASES: Dict[str, str] = {
    "openai_compatible": _OPENAI_COMPATIBLE_SPEC,
    "openai-compatible": _OPENAI_COMPATIBLE_SPEC,
    "openai": _OPENAI_COMPATIBLE_SPEC,
    "deepseek": _OPENAI_COMPATIBLE_SPEC,
    "deep_seek": _OPENAI_COMPATIBLE_SPEC,
    "openrouter": _OPENAI_COMPATIBLE_SPEC,
    "siliconflow": _OPENAI_COMPATIBLE_SPEC,
    "moonshot": _OPENAI_COMPATIBLE_SPEC,
    "kimi": _OPENAI_COMPATIBLE_SPEC,
    "qwen": _OPENAI_COMPATIBLE_SPEC,
}


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout_s: int = 60


def load_llm_settings_from_env(prefix: str = "RETROLLM_LLM_") -> Optional[LLMSettings]:
    provider = os.environ.get(prefix + "PROVIDER")
    if not provider:
        return None
    base_url = os.environ.get(prefix + "BASE_URL", "").strip()
    api_key = os.environ.get(prefix + "API_KEY", "").strip()
    model = os.environ.get(prefix + "MODEL", "").strip()
    temperature = float(os.environ.get(prefix + "TEMPERATURE", "0.2"))
    timeout_s = int(os.environ.get(prefix + "TIMEOUT_S", "60"))
    return LLMSettings(
        provider=provider.strip(),
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
    )


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower()


def _resolve_provider_spec(settings: LLMSettings) -> str:
    provider = settings.provider.strip()
    normalized = _normalize_provider(provider)
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    if ":" in provider:
        return provider
    if settings.base_url:
        # For any plain provider name with BASE_URL set, assume OpenAI-compatible mode.
        return _OPENAI_COMPATIBLE_SPEC
    raise ValueError(
        "Unknown provider alias "
        f"'{provider}'. Use RETROLLM_LLM_PROVIDER=openai_compatible "
        "(or deepseek/openrouter/etc.) for OpenAI-compatible endpoints, "
        "or a class path 'package.module:ClassName'."
    )


def _import_from_spec(spec: str) -> Type[Any]:
    if ":" not in spec:
        raise ValueError(
            "Provider must be an alias or a class path 'package.module:ClassName'"
        )
    module_name, cls_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls


def build_provider(settings: LLMSettings) -> LLMProvider:
    spec = _resolve_provider_spec(settings)
    cls = _import_from_spec(spec)

    if cls is OpenAICompatibleProvider:
        if not settings.base_url or not settings.api_key:
            raise ValueError(
                "OpenAI-compatible provider requires RETROLLM_LLM_BASE_URL and RETROLLM_LLM_API_KEY"
            )
        return OpenAICompatibleProvider(base_url=settings.base_url, api_key=settings.api_key)

    provider = cls()  # type: ignore[call-arg]
    if not isinstance(provider, LLMProvider):
        raise TypeError("Custom provider must subclass LLMProvider")
    return provider
