"""High-level planner API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Any

from retrollm.chem import Molecule
from retrollm.config import ArtifactsConfig, load_artifacts_config
from retrollm.llm.controller import LLMMetaController
from retrollm.policy.onnx_filter import OnnxFilterPolicy
from retrollm.policy.onnx_policy import OnnxTemplatePolicy
from retrollm.policy.reaction_engine import ReactionEngine
from retrollm.policy.templates import TemplateLibrary
from retrollm.search import SearchConfig, SearchResult, SearchTree
from retrollm.stock import InchiKeyStock


@dataclass
class Planner:
    """Standalone planner orchestrating stock, policy, and search."""

    artifacts: ArtifactsConfig
    search_config: SearchConfig

    def __post_init__(self) -> None:
        self.stock = InchiKeyStock(self.artifacts.stock)
        self.policy = OnnxTemplatePolicy(self.artifacts.policy_model_onnx)
        self.filter_policy = OnnxFilterPolicy(self.artifacts.filter_policy_onnx)
        self.templates = TemplateLibrary(self.artifacts.template_file)
        self.ringbreaker_policy = OnnxTemplatePolicy(self.artifacts.ringbreaker_model_onnx)
        self.ringbreaker_templates = TemplateLibrary(self.artifacts.ringbreaker_templates)
        self.reaction_engine = ReactionEngine()

    @classmethod
    def from_config_file(
        cls, config_file: str, search_config: SearchConfig | None = None
    ) -> "Planner":
        artifacts = load_artifacts_config(config_file)
        return cls(artifacts=artifacts, search_config=search_config or SearchConfig())

    def search(self, target_smiles: str, use_llm: bool = False) -> SearchResult:
        llm_controller = LLMMetaController.from_env() if use_llm else None
        constraints = self._resolve_constraints(llm_controller)
        result = self._run_search_once(
            target_smiles=target_smiles,
            config=self.search_config,
            llm_controller=llm_controller,
            constraints=constraints,
        )

        retry_attempts = 0
        if (
            llm_controller is not None
            and self.search_config.llm_use_failure_diagnosis
            and not result.solved
            and hasattr(llm_controller, "diagnose_failure")
        ):
            diagnosis = llm_controller.diagnose_failure(
                diagnostics=result.llm.get("final_diagnostics", {}),
                constraints=constraints,
                config_snapshot=asdict(self.search_config),
            )
            result.llm["diagnosis"] = diagnosis

            if self.search_config.llm_retry_on_failure:
                current = result
                current_config = self.search_config
                for _ in range(max(0, self.search_config.llm_max_retry_attempts)):
                    retry_plan = diagnosis.get("retry_plan", {})
                    next_config = self._apply_retry_plan(current_config, retry_plan)
                    if next_config == current_config:
                        break

                    retry_attempts += 1
                    retry_result = self._run_search_once(
                        target_smiles=target_smiles,
                        config=next_config,
                        llm_controller=llm_controller,
                        constraints=constraints,
                    )
                    retry_result.retry_attempts = retry_attempts
                    retry_result.llm["retry_plan_applied"] = retry_plan
                    retry_result.llm["retry_from_config"] = asdict(current_config)
                    retry_result.llm["diagnosis"] = diagnosis

                    if self._is_better_result(retry_result, current):
                        current = retry_result
                    current_config = next_config
                    if current.solved:
                        break
                result = current

        result.retry_attempts = retry_attempts
        if llm_controller is not None and self.search_config.llm_use_handoff:
            handoff = llm_controller.generate_handoff(
                target_smiles=target_smiles,
                routes=result.routes,
                constraints=constraints,
                diagnosis=result.llm.get("diagnosis"),
            )
            result.llm["handoff_markdown"] = handoff
        return result

    def _run_search_once(
        self,
        target_smiles: str,
        config: SearchConfig,
        llm_controller: LLMMetaController | None,
        constraints: dict[str, Any],
    ) -> SearchResult:
        tree = SearchTree(
            target=Molecule(target_smiles),
            stock=self.stock,
            policy=self.policy,
            templates=self.templates,
            reaction_engine=self.reaction_engine,
            config=config,
            llm_controller=llm_controller,
            filter_policy=self.filter_policy,
            ringbreaker_policy=self.ringbreaker_policy,
            ringbreaker_templates=self.ringbreaker_templates,
            constraints=constraints,
        )
        return tree.run()

    def _resolve_constraints(
        self, llm_controller: LLMMetaController | None
    ) -> dict[str, Any]:
        text = self.search_config.constraints_text.strip()
        if not text:
            return {}
        if llm_controller is None:
            return self._heuristic_constraints_from_text(text)
        return llm_controller.translate_constraints(text)

    def _apply_retry_plan(
        self, config: SearchConfig, retry_plan: dict[str, Any]
    ) -> SearchConfig:
        if not retry_plan:
            return config

        new_config = replace(config)
        for key, value in retry_plan.items():
            if not hasattr(new_config, key):
                continue
            setattr(new_config, key, value)
        return new_config

    def _is_better_result(self, candidate: SearchResult, baseline: SearchResult) -> bool:
        if candidate.solved and not baseline.solved:
            return True
        if baseline.solved and not candidate.solved:
            return False
        if len(candidate.routes) != len(baseline.routes):
            return len(candidate.routes) > len(baseline.routes)
        if candidate.root_children != baseline.root_children:
            return candidate.root_children > baseline.root_children
        return candidate.iterations >= baseline.iterations

    def _heuristic_constraints_from_text(self, text: str) -> dict[str, Any]:
        lowered = text.lower()
        step_match = re.search(r"(\d+)\s*steps?", lowered)
        max_steps: int | None = int(step_match.group(1)) if step_match else None

        avoid_reactants: list[str] = []
        for chunk in re.findall(r"avoid\s+([a-z0-9,\s\-\+_/]+)", lowered):
            for token in chunk.split(","):
                token = token.strip()
                if token and token not in {"and", "or"}:
                    avoid_reactants.append(token)

        avoid_template_indices = [
            int(value) for value in re.findall(r"template\s+(\d+)", lowered)
        ]
        prefer_in_stock_subgoals = any(
            keyword in lowered
            for keyword in ("in stock", "commercially available", "purchasable")
        )
        return {
            "source": "heuristic_without_llm",
            "avoid_reactants": avoid_reactants,
            "avoid_template_indices": avoid_template_indices,
            "max_steps": max_steps,
            "prefer_in_stock_subgoals": prefer_in_stock_subgoals,
            "notes": text,
        }
