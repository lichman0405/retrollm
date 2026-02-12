"""High-level planner API."""

from __future__ import annotations

from dataclasses import dataclass

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
        tree = SearchTree(
            target=Molecule(target_smiles),
            stock=self.stock,
            policy=self.policy,
            templates=self.templates,
            reaction_engine=self.reaction_engine,
            config=self.search_config,
            llm_controller=llm_controller,
            filter_policy=self.filter_policy,
            ringbreaker_policy=self.ringbreaker_policy,
            ringbreaker_templates=self.ringbreaker_templates,
        )
        return tree.run()
