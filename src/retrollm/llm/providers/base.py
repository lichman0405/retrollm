"""LLM provider interface.

Providers must implement `chat()` and return the assistant text.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.2,
        timeout_s: int = 60,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError
