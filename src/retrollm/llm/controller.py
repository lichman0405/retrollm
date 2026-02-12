"""LLM meta-controller for adaptive search parameters."""

from __future__ import annotations

import json

from retrollm.llm.provider_loader import (
    LLMSettings,
    build_provider,
    load_llm_settings_from_env,
)
from retrollm.llm.providers.base import ChatMessage


SYSTEM_PROMPT = (
    "You are a retrosynthesis MCTS optimizer. "
    "Always respond with strict JSON and adjust only one parameter."
)


class LLMMetaController:
    """A lightweight adaptive controller that can tune search hyperparameters."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.provider = build_provider(settings)

    @classmethod
    def from_env(cls) -> "LLMMetaController | None":
        settings = load_llm_settings_from_env()
        if settings is None:
            return None
        if not settings.model:
            raise ValueError("RETROLLM_LLM_MODEL is required when LLM provider is configured")
        return cls(settings)

    def maybe_adjust(self, tree: "SearchTree") -> None:
        diagnostics = tree.diagnostics()
        if diagnostics["first_solution_iteration"] is not None:
            return

        user_prompt = (
            "Search diagnostics:\n"
            f"{json.dumps(diagnostics, ensure_ascii=False)}\n\n"
            "Return strict JSON with keys: action, parameter, value, reason.\n"
            "Allowed parameter values: c_puct, topk_templates.\n"
            "Action must be one of: increase, decrease, keep."
        )
        text = self.provider.chat(
            messages=[
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_prompt),
            ],
            model=self.settings.model,
            temperature=self.settings.temperature,
            timeout_s=self.settings.timeout_s,
        )
        adjustment = self._safe_parse(text)
        self._apply_adjustment(tree, adjustment)

    def _safe_parse(self, text: str) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < 0 or end <= start:
                return {"action": "keep"}
            return json.loads(text[start : end + 1])
        except Exception:
            return {"action": "keep"}

    def _apply_adjustment(self, tree: "SearchTree", adjustment: dict) -> None:
        action = str(adjustment.get("action", "keep")).lower()
        parameter = str(adjustment.get("parameter", ""))
        value = adjustment.get("value")

        if action == "keep":
            return

        if parameter == "c_puct":
            try:
                new_value = float(value)
            except Exception:
                return
            tree.config.c_puct = min(5.0, max(0.1, new_value))
            return

        if parameter == "topk_templates":
            try:
                new_value = int(value)
            except Exception:
                return
            tree.config.topk_templates = min(200, max(5, new_value))


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrollm.search.tree import SearchTree
