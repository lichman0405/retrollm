from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("rdkit")

from retrollm.chem import Molecule
from retrollm.policy.onnx_policy import TemplatePrediction
from retrollm.search import SearchConfig, SearchTree


class _DummyStock:
    def __init__(self, in_stock_smiles: set[str]):
        self._in_stock_smiles = in_stock_smiles

    def __contains__(self, mol: Molecule) -> bool:
        return mol.smiles in self._in_stock_smiles


class _DummyPolicy:
    def __init__(self, predictions: list[TemplatePrediction]):
        self._predictions = predictions

    def predict_topk(self, fingerprint: np.ndarray, topk: int = 50) -> list[TemplatePrediction]:
        del fingerprint
        return self._predictions[:topk]


class _DummyTemplates:
    def __init__(self, mapping: dict[int, str]):
        self._mapping = mapping

    def smarts_by_index(self, index: int) -> str:
        return self._mapping[index]


@dataclass(frozen=True)
class _DummyOutcome:
    reactants: tuple[Molecule, ...]


class _DummyReactionEngine:
    def __init__(self, outcomes: dict[str, list[_DummyOutcome]]):
        self._outcomes = outcomes

    def apply(self, product: Molecule, smarts: str) -> list[_DummyOutcome]:
        del product
        return self._outcomes.get(smarts, [])


class _DummyFilter:
    def __init__(self, score: float):
        self.score = score

    def score_reaction(self, product: Molecule, reactants: tuple[Molecule, ...]) -> float:
        del product, reactants
        return self.score


class _DummyLLM:
    def __init__(self):
        self._events: list[dict] = []

    def maybe_adjust(self, tree: SearchTree) -> None:
        del tree
        self._events.append({"stage": "meta_control", "message": "noop"})

    def choose_expansion_target(self, expandable_smiles: list[str], context: dict) -> dict:
        del context
        if len(expandable_smiles) >= 2:
            return {"index": 1, "reason": "choose_second"}
        return {"index": 0, "reason": "single"}

    def rerank_routes(self, routes: list[dict], objective: dict) -> tuple[list[dict], dict]:
        del objective
        ranked = list(reversed([dict(route) for route in routes]))
        for idx, route in enumerate(ranked, start=1):
            route["llm_rank"] = idx
            route["llm_rank_score"] = 1.0 - (idx * 0.1)
            route["llm_rank_reason"] = "dummy"
        self._events.append({"stage": "rerank", "message": "applied"})
        return ranked, {"applied": True, "mode": "llm", "global_reason": "dummy"}

    def events_as_dict(self) -> list[dict]:
        return list(self._events)


def test_ringbreaker_fallback_populates_routes() -> None:
    stock = _DummyStock({"CCO", "CCN"})
    base_policy = _DummyPolicy([TemplatePrediction(index=0, probability=0.8)])
    ringbreaker_policy = _DummyPolicy([TemplatePrediction(index=0, probability=0.6)])
    templates = _DummyTemplates({0: "base"})
    ringbreaker_templates = _DummyTemplates({0: "ringbreaker"})
    reaction_engine = _DummyReactionEngine(
        {
            "ringbreaker": [_DummyOutcome((Molecule("CCO"), Molecule("CCN")))],
        }
    )
    tree = SearchTree(
        target=Molecule("CCOC"),
        stock=stock,  # type: ignore[arg-type]
        policy=base_policy,  # type: ignore[arg-type]
        templates=templates,  # type: ignore[arg-type]
        reaction_engine=reaction_engine,  # type: ignore[arg-type]
        config=SearchConfig(
            iteration_limit=6,
            topk_templates=1,
            ringbreaker_topk=1,
            max_routes=2,
            use_filter=False,
            use_ringbreaker=True,
        ),
        ringbreaker_policy=ringbreaker_policy,  # type: ignore[arg-type]
        ringbreaker_templates=ringbreaker_templates,  # type: ignore[arg-type]
    )
    result = tree.run()

    assert result.solved is True
    assert result.root_children >= 1
    assert result.routes
    assert result.routes[0]["steps"][0]["policy_source"] == "ringbreaker"


def test_filter_cutoff_blocks_expansion() -> None:
    stock = _DummyStock({"CCO", "CCN"})
    base_policy = _DummyPolicy([TemplatePrediction(index=0, probability=0.9)])
    templates = _DummyTemplates({0: "base"})
    reaction_engine = _DummyReactionEngine(
        {
            "base": [_DummyOutcome((Molecule("CCO"), Molecule("CCN")))],
        }
    )
    tree = SearchTree(
        target=Molecule("CCOC"),
        stock=stock,  # type: ignore[arg-type]
        policy=base_policy,  # type: ignore[arg-type]
        templates=templates,  # type: ignore[arg-type]
        reaction_engine=reaction_engine,  # type: ignore[arg-type]
        config=SearchConfig(
            iteration_limit=3,
            topk_templates=1,
            use_filter=True,
            filter_cutoff=0.5,
            use_ringbreaker=False,
        ),
        filter_policy=_DummyFilter(score=0.1),  # type: ignore[arg-type]
    )
    result = tree.run()

    assert result.root_children == 0
    assert result.routes == []


def test_constraint_avoid_reactants_blocks_expansion() -> None:
    stock = _DummyStock({"CCO", "CCN"})
    base_policy = _DummyPolicy([TemplatePrediction(index=0, probability=0.9)])
    templates = _DummyTemplates({0: "base"})
    reaction_engine = _DummyReactionEngine(
        {
            "base": [_DummyOutcome((Molecule("CCO"), Molecule("NCCNCCN")))],
        }
    )
    tree = SearchTree(
        target=Molecule("CCOC"),
        stock=stock,  # type: ignore[arg-type]
        policy=base_policy,  # type: ignore[arg-type]
        templates=templates,  # type: ignore[arg-type]
        reaction_engine=reaction_engine,  # type: ignore[arg-type]
        config=SearchConfig(
            iteration_limit=3,
            topk_templates=1,
            use_filter=False,
            use_ringbreaker=False,
        ),
        constraints={"avoid_reactants": ["NCCNCCN"]},
    )
    result = tree.run()

    assert result.root_children == 0
    assert result.llm["constraint_drop_count"] >= 1


def test_llm_rerank_metadata_is_recorded() -> None:
    stock = _DummyStock(set())
    base_policy = _DummyPolicy(
        [
            TemplatePrediction(index=0, probability=0.9),
            TemplatePrediction(index=1, probability=0.8),
        ]
    )
    templates = _DummyTemplates({0: "a", 1: "b"})
    reaction_engine = _DummyReactionEngine(
        {
            "a": [_DummyOutcome((Molecule("CCO"),))],
            "b": [_DummyOutcome((Molecule("CCC"),))],
        }
    )
    llm = _DummyLLM()
    tree = SearchTree(
        target=Molecule("CCOC"),
        stock=stock,  # type: ignore[arg-type]
        policy=base_policy,  # type: ignore[arg-type]
        templates=templates,  # type: ignore[arg-type]
        reaction_engine=reaction_engine,  # type: ignore[arg-type]
        config=SearchConfig(
            iteration_limit=2,
            topk_templates=2,
            use_filter=False,
            use_ringbreaker=False,
            max_routes=2,
            llm_use_route_reranker=True,
        ),
        llm_controller=llm,  # type: ignore[arg-type]
    )
    result = tree.run()

    assert result.llm["rerank"]["applied"] is True
    assert result.routes
    assert result.routes[0]["llm_rank"] == 1
