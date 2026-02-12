"""LLM workflow controller for retrosynthesis search.

Responsibilities:
- Online meta-control (MCTS parameter adjustment)
- Natural-language constraint translation
- In-search subgoal advisor
- Post-search route reranking
- Failure diagnosis and retry-plan suggestion
- Final route explanation/handoff draft
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from retrollm.llm.provider_loader import (
    LLMSettings,
    build_provider,
    load_llm_settings_from_env,
)
from retrollm.llm.providers.base import ChatMessage


SYSTEM_PROMPT = (
    "You are a retrosynthesis MCTS optimizer. "
    "Always respond with strict JSON when asked for JSON."
)


@dataclass
class LLMEvent:
    stage: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "payload": self.payload,
        }


class LLMMetaController:
    """Unified LLM workflow controller."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.provider = build_provider(settings)
        self.events: list[LLMEvent] = []

    @classmethod
    def from_env(cls) -> "LLMMetaController | None":
        settings = load_llm_settings_from_env()
        if settings is None:
            return None
        if not settings.model:
            raise ValueError("RETROLLM_LLM_MODEL is required when LLM provider is configured")
        return cls(settings)

    def _emit(self, stage: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append(LLMEvent(stage=stage, message=message, payload=payload or {}))

    def events_as_dict(self) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self.events]

    def maybe_adjust(self, tree: "SearchTree") -> None:
        diagnostics = tree.diagnostics()
        if diagnostics["first_solution_iteration"] is not None:
            self._emit("meta_control", "skip_adjustment_solution_exists")
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
        self._emit(
            "meta_control",
            "adjustment_applied",
            {
                "input_diagnostics": diagnostics,
                "adjustment": adjustment,
                "c_puct": tree.config.c_puct,
                "topk_templates": tree.config.topk_templates,
            },
        )

    def translate_constraints(self, constraint_text: str) -> dict[str, Any]:
        text = constraint_text.strip()
        if not text:
            payload = self._sanitize_constraints({})
            payload["source"] = "empty"
            self._emit("constraint_translation", "no_constraints_provided", payload)
            return payload

        user_prompt = (
            "Translate user constraints for retrosynthesis into strict JSON with keys:\n"
            "- avoid_reactants: string[]\n"
            "- avoid_template_indices: int[]\n"
            "- max_steps: int|null\n"
            "- prefer_in_stock_subgoals: bool\n"
            "- notes: string\n"
            f"\nUser constraints:\n{text}"
        )
        try:
            response = self.provider.chat(
                messages=[
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=user_prompt),
                ],
                model=self.settings.model,
                temperature=self.settings.temperature,
                timeout_s=self.settings.timeout_s,
            )
            parsed = self._sanitize_constraints(self._safe_parse(response))
            parsed["source"] = "llm"
            self._emit("constraint_translation", "translated_with_llm", parsed)
            return parsed
        except Exception as exc:  # noqa: BLE001
            fallback = self._heuristic_constraints(text)
            fallback["source"] = "heuristic_fallback"
            fallback["error"] = str(exc)
            self._emit("constraint_translation", "translated_with_fallback", fallback)
            return fallback

    def choose_expansion_target(
        self, expandable_smiles: list[str], context: dict[str, Any]
    ) -> dict[str, Any]:
        if not expandable_smiles:
            return {"index": 0, "reason": "empty_candidate_list"}
        if len(expandable_smiles) == 1:
            return {"index": 0, "reason": "single_candidate"}

        user_prompt = (
            "Choose which molecule should be expanded first in MCTS.\n"
            "Return strict JSON with keys: index (int), reason (string).\n"
            f"Candidates: {json.dumps(expandable_smiles, ensure_ascii=False)}\n"
            f"Context: {json.dumps(context, ensure_ascii=False)}"
        )
        try:
            response = self.provider.chat(
                messages=[
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=user_prompt),
                ],
                model=self.settings.model,
                temperature=self.settings.temperature,
                timeout_s=self.settings.timeout_s,
            )
            data = self._safe_parse(response)
            idx = int(data.get("index", 0))
            idx = max(0, min(len(expandable_smiles) - 1, idx))
            choice = {"index": idx, "reason": str(data.get("reason", "")).strip()}
            self._emit("subgoal_advisor", "llm_choice", choice)
            return choice
        except Exception as exc:  # noqa: BLE001
            idx = max(range(len(expandable_smiles)), key=lambda i: len(expandable_smiles[i]))
            choice = {
                "index": idx,
                "reason": "heuristic_longest_smiles",
                "error": str(exc),
            }
            self._emit("subgoal_advisor", "fallback_choice", choice)
            return choice

    def rerank_routes(
        self, routes: list[dict[str, Any]], objective: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not routes:
            return routes, {"applied": False, "reason": "no_routes"}

        compact_routes: list[dict[str, Any]] = []
        for idx, route in enumerate(routes, start=1):
            compact_routes.append(
                {
                    "route_index": idx,
                    "solved": bool(route.get("solved", False)),
                    "score": float(route.get("score", 0.0)),
                    "depth": int(route.get("depth", 0)),
                    "steps": len(route.get("steps", [])),
                    "terminal_molecules": route.get("molecules", []),
                }
            )
        user_prompt = (
            "Rerank candidate retrosynthesis routes.\n"
            "Return strict JSON with keys:\n"
            "- ranking: [{route_index:int, score:float, reason:string}]\n"
            "- global_reason: string\n"
            f"Objective: {json.dumps(objective, ensure_ascii=False)}\n"
            f"Routes: {json.dumps(compact_routes, ensure_ascii=False)}"
        )

        try:
            response = self.provider.chat(
                messages=[
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=user_prompt),
                ],
                model=self.settings.model,
                temperature=self.settings.temperature,
                timeout_s=self.settings.timeout_s,
            )
            parsed = self._safe_parse(response)
            ranking = parsed.get("ranking", [])
            score_by_index: dict[int, tuple[float, str]] = {}
            if isinstance(ranking, list):
                for item in ranking:
                    if not isinstance(item, dict):
                        continue
                    try:
                        route_idx = int(item.get("route_index", -1))
                        if route_idx < 1 or route_idx > len(routes):
                            continue
                        raw_score = float(item.get("score", 0.0))
                        score = min(1.0, max(0.0, raw_score))
                        reason = str(item.get("reason", "")).strip()
                        score_by_index[route_idx] = (score, reason)
                    except Exception:
                        continue

            if not score_by_index:
                raise ValueError("Invalid or empty LLM ranking")

            scored: list[dict[str, Any]] = []
            for idx, route in enumerate(routes, start=1):
                score, reason = score_by_index.get(idx, (0.0, "not_ranked_by_llm"))
                enriched = dict(route)
                enriched["llm_rank_score"] = score
                enriched["llm_rank_reason"] = reason
                scored.append(enriched)
            ranked = sorted(
                scored,
                key=lambda route: (
                    float(route.get("llm_rank_score", 0.0)),
                    float(route.get("score", 0.0)),
                    -int(route.get("depth", 0)),
                ),
                reverse=True,
            )
            for rank, route in enumerate(ranked, start=1):
                route["llm_rank"] = rank

            meta = {
                "applied": True,
                "mode": "llm",
                "global_reason": str(parsed.get("global_reason", "")).strip(),
            }
            self._emit("route_rerank", "rerank_with_llm", meta)
            return ranked, meta
        except Exception as exc:  # noqa: BLE001
            ranked = self._heuristic_rerank(routes)
            meta = {
                "applied": True,
                "mode": "heuristic_fallback",
                "global_reason": "LLM rerank failed, used heuristic ordering",
                "error": str(exc),
            }
            self._emit("route_rerank", "rerank_with_fallback", meta)
            return ranked, meta

    def diagnose_failure(
        self,
        diagnostics: dict[str, Any],
        constraints: dict[str, Any],
        config_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        user_prompt = (
            "Diagnose a retrosynthesis search failure and provide one retry plan.\n"
            "Return strict JSON with keys:\n"
            "- diagnosis: string\n"
            "- retry_plan: object containing optional keys:\n"
            "  iteration_limit, time_limit_s, c_puct, topk_templates, max_depth,\n"
            "  filter_cutoff, use_filter, use_ringbreaker, ringbreaker_topk\n"
            "- rationale: string\n"
            f"\nDiagnostics: {json.dumps(diagnostics, ensure_ascii=False)}"
            f"\nConstraints: {json.dumps(constraints, ensure_ascii=False)}"
            f"\nConfig: {json.dumps(config_snapshot, ensure_ascii=False)}"
        )
        try:
            response = self.provider.chat(
                messages=[
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=user_prompt),
                ],
                model=self.settings.model,
                temperature=self.settings.temperature,
                timeout_s=self.settings.timeout_s,
            )
            parsed = self._safe_parse(response)
            diagnosis = {
                "diagnosis": str(parsed.get("diagnosis", "")).strip(),
                "retry_plan": self._sanitize_retry_plan(parsed.get("retry_plan", {})),
                "rationale": str(parsed.get("rationale", "")).strip(),
                "source": "llm",
            }
            self._emit("failure_diagnosis", "diagnosis_with_llm", diagnosis)
            return diagnosis
        except Exception as exc:  # noqa: BLE001
            fallback = self._heuristic_failure_plan(diagnostics, config_snapshot)
            fallback["source"] = "heuristic_fallback"
            fallback["error"] = str(exc)
            self._emit("failure_diagnosis", "diagnosis_with_fallback", fallback)
            return fallback

    def generate_handoff(
        self,
        target_smiles: str,
        routes: list[dict[str, Any]],
        constraints: dict[str, Any],
        diagnosis: dict[str, Any] | None,
        no_route_reason: str | None = None,
    ) -> str:
        if not routes:
            reason_text = {
                "target_in_stock": "Target molecule is already in stock.",
                "all_expansions_filtered_by_constraints": "All candidate expansions were filtered by constraints.",
                "filter_runtime_error": "Reaction filter failed at runtime and no valid expansions remained.",
                "no_valid_expansions": "No valid expansion was produced from available templates.",
                "no_routes_after_ranking": "Routes were removed during ranking/selection.",
            }.get(str(no_route_reason), "No valid route candidates were found.")
            return (
                "# RetroLLM Handoff\n\n"
                f"- Target: `{target_smiles}`\n"
                "- No candidate routes available.\n"
                f"- Reason: {reason_text}\n"
            )

        route_preview = []
        for idx, route in enumerate(routes[:3], start=1):
            route_preview.append(
                {
                    "route_index": idx,
                    "solved": route.get("solved"),
                    "score": route.get("score"),
                    "depth": route.get("depth"),
                    "steps": route.get("steps", []),
                }
            )

        user_prompt = (
            "Write a concise markdown handoff for chemistry team.\n"
            "Sections required:\n"
            "- Objective\n"
            "- Route Summary\n"
            "- Key Risks\n"
            "- Suggested Next Experiments\n"
            f"\nTarget: {target_smiles}"
            f"\nConstraints: {json.dumps(constraints, ensure_ascii=False)}"
            f"\nDiagnosis: {json.dumps(diagnosis or {}, ensure_ascii=False)}"
            f"\nRoutes: {json.dumps(route_preview, ensure_ascii=False)}"
        )
        try:
            text = self.provider.chat(
                messages=[
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=user_prompt),
                ],
                model=self.settings.model,
                temperature=self.settings.temperature,
                timeout_s=self.settings.timeout_s,
            )
            output = text.strip()
            if not output:
                raise ValueError("Empty handoff response")
            self._emit("handoff", "handoff_generated_with_llm", {"length": len(output)})
            return output
        except Exception as exc:  # noqa: BLE001
            fallback = self._heuristic_handoff(target_smiles, routes, diagnosis)
            self._emit(
                "handoff",
                "handoff_generated_with_fallback",
                {"error": str(exc), "length": len(fallback)},
            )
            return fallback

    def _safe_parse(self, text: str) -> dict[str, Any]:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < 0 or end <= start:
                return {}
            return json.loads(text[start : end + 1])
        except Exception:
            return {}

    def _apply_adjustment(self, tree: "SearchTree", adjustment: dict[str, Any]) -> None:
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

    def _sanitize_constraints(self, payload: dict[str, Any]) -> dict[str, Any]:
        avoid_reactants_raw = payload.get("avoid_reactants", [])
        avoid_templates_raw = payload.get("avoid_template_indices", [])
        max_steps_raw = payload.get("max_steps")
        prefer_stock_raw = payload.get("prefer_in_stock_subgoals", False)
        notes_raw = payload.get("notes", "")

        avoid_reactants: list[str] = []
        if isinstance(avoid_reactants_raw, list):
            for item in avoid_reactants_raw:
                s = str(item).strip()
                if s:
                    avoid_reactants.append(s)

        avoid_templates: list[int] = []
        if isinstance(avoid_templates_raw, list):
            for item in avoid_templates_raw:
                try:
                    avoid_templates.append(int(item))
                except Exception:
                    continue

        max_steps: int | None = None
        if max_steps_raw is not None:
            try:
                max_steps = int(max_steps_raw)
                if max_steps <= 0:
                    max_steps = None
            except Exception:
                max_steps = None

        return {
            "avoid_reactants": avoid_reactants,
            "avoid_template_indices": avoid_templates,
            "max_steps": max_steps,
            "prefer_in_stock_subgoals": bool(prefer_stock_raw),
            "notes": str(notes_raw).strip(),
        }

    def _heuristic_constraints(self, text: str) -> dict[str, Any]:
        lowered = text.lower()
        step_match = re.search(r"(\d+)\s*steps?", lowered)
        max_steps: int | None = int(step_match.group(1)) if step_match else None

        avoid_reactants: list[str] = []
        avoid_chunks = re.findall(r"avoid\s+([a-z0-9,\s\-\+_/]+)", lowered)
        for chunk in avoid_chunks:
            for token in chunk.split(","):
                token = token.strip()
                if token and token not in {"and", "or"}:
                    avoid_reactants.append(token)

        avoid_template_indices: list[int] = []
        for match in re.findall(r"template\s+(\d+)", lowered):
            avoid_template_indices.append(int(match))

        prefer_in_stock = any(
            keyword in lowered
            for keyword in ("in stock", "commercially available", "purchasable")
        )
        return self._sanitize_constraints(
            {
                "avoid_reactants": avoid_reactants,
                "avoid_template_indices": avoid_template_indices,
                "max_steps": max_steps,
                "prefer_in_stock_subgoals": prefer_in_stock,
                "notes": text.strip(),
            }
        )

    def _heuristic_rerank(self, routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched = []
        for route in routes:
            solved_bonus = 0.3 if bool(route.get("solved")) else 0.0
            score = float(route.get("score", 0.0))
            depth_penalty = min(float(route.get("depth", 0)), 10.0) * 0.03
            rank_score = max(0.0, min(1.0, score + solved_bonus - depth_penalty))
            copy = dict(route)
            copy["llm_rank_score"] = rank_score
            copy["llm_rank_reason"] = "heuristic_score_solved_depth"
            enriched.append(copy)
        ranked = sorted(
            enriched,
            key=lambda route: (
                float(route.get("llm_rank_score", 0.0)),
                float(route.get("score", 0.0)),
                -int(route.get("depth", 0)),
            ),
            reverse=True,
        )
        for rank, route in enumerate(ranked, start=1):
            route["llm_rank"] = rank
        return ranked

    def _sanitize_retry_plan(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        out: dict[str, Any] = {}

        def _try_int(key: str, lo: int, hi: int) -> None:
            if key not in payload:
                return
            try:
                val = int(payload[key])
            except Exception:
                return
            out[key] = min(hi, max(lo, val))

        def _try_float(key: str, lo: float, hi: float) -> None:
            if key not in payload:
                return
            try:
                val = float(payload[key])
            except Exception:
                return
            out[key] = min(hi, max(lo, val))

        def _try_bool(key: str) -> None:
            if key in payload:
                out[key] = bool(payload[key])

        _try_int("iteration_limit", 20, 5000)
        _try_float("time_limit_s", 10.0, 3600.0)
        _try_float("c_puct", 0.1, 5.0)
        _try_int("topk_templates", 5, 300)
        _try_int("max_depth", 1, 12)
        _try_float("filter_cutoff", 0.0, 1.0)
        _try_bool("use_filter")
        _try_bool("use_ringbreaker")
        _try_int("ringbreaker_topk", 1, 200)
        return out

    def _heuristic_failure_plan(
        self, diagnostics: dict[str, Any], config_snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        root_children = int(diagnostics.get("root_children", 0))
        topk = int(config_snapshot.get("topk_templates", 20))
        max_depth = int(config_snapshot.get("max_depth", 6))
        iteration_limit = int(config_snapshot.get("iteration_limit", 150))
        use_filter = bool(config_snapshot.get("use_filter", True))
        ringbreaker_topk = int(config_snapshot.get("ringbreaker_topk", 10))
        time_limit_s = float(config_snapshot.get("time_limit_s", 120.0))
        c_puct = float(config_snapshot.get("c_puct", 1.4))

        retry_plan: dict[str, Any] = {}
        diagnosis = "Search failed to find a solved route."
        rationale = "Use broader exploration and less restrictive filtering."

        if root_children == 0:
            diagnosis = "No children were expanded at root; expansion may be over-restricted."
            retry_plan["topk_templates"] = min(300, max(30, topk * 2))
            retry_plan["max_depth"] = min(12, max_depth + 2)
            if use_filter:
                retry_plan["use_filter"] = False
            retry_plan["ringbreaker_topk"] = min(200, max(20, ringbreaker_topk * 2))
        else:
            retry_plan["iteration_limit"] = min(5000, max(300, int(iteration_limit * 1.8)))
            retry_plan["time_limit_s"] = min(3600.0, max(180.0, time_limit_s * 1.5))
            retry_plan["c_puct"] = min(5.0, max(0.5, c_puct * 0.85))
        return {
            "diagnosis": diagnosis,
            "retry_plan": retry_plan,
            "rationale": rationale,
        }

    def _heuristic_handoff(
        self,
        target_smiles: str,
        routes: list[dict[str, Any]],
        diagnosis: dict[str, Any] | None,
    ) -> str:
        lines = [
            "# RetroLLM Handoff",
            "",
            "## Objective",
            f"- Target molecule: `{target_smiles}`",
            "",
            "## Route Summary",
        ]
        for idx, route in enumerate(routes[:3], start=1):
            lines.append(
                f"- Route {idx}: solved={route.get('solved')} "
                f"score={route.get('score')} depth={route.get('depth')} "
                f"steps={len(route.get('steps', []))}"
            )
        if diagnosis:
            lines.extend(
                [
                    "",
                    "## Failure Analysis / Retry Context",
                    f"- Diagnosis: {diagnosis.get('diagnosis', '')}",
                    f"- Suggested retry plan: `{diagnosis.get('retry_plan', {})}`",
                ]
            )
        lines.extend(
            [
                "",
                "## Key Risks",
                "- Template-based routes may include low-confidence transformations.",
                "- Validate reagent availability and reaction feasibility before lab execution.",
                "",
                "## Suggested Next Experiments",
                "- Prioritize routes with higher rank score and fewer high-risk reagents.",
                "- Run single-step validation for the first disconnection before full synthesis.",
            ]
        )
        return "\n".join(lines)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrollm.search.tree import SearchTree
