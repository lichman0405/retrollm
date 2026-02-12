"""OpenAI-compatible HTTP provider.

This supports any endpoint that implements `POST {base_url}/v1/chat/completions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from retrollm.llm.providers.base import ChatMessage, LLMProvider


@dataclass(frozen=True)
class OpenAICompatibleProvider(LLMProvider):
    base_url: str
    api_key: str

    def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.2,
        timeout_s: int = 60,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if extra:
            payload.update(extra)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected OpenAI-compatible response shape: {data}") from exc
