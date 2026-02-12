"""MCTS search tree implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time

from retrollm.chem import Molecule
from retrollm.llm.controller import LLMMetaController
from retrollm.policy.onnx_filter import OnnxFilterPolicy
from retrollm.policy.onnx_policy import OnnxTemplatePolicy
from retrollm.policy.reaction_engine import ReactionEngine
from retrollm.policy.templates import TemplateLibrary
from retrollm.search.node import NodeTransition, SearchNode
from retrollm.search.state import SearchState
from retrollm.stock import InchiKeyStock


@dataclass
class SearchConfig:
    iteration_limit: int = 150
    time_limit_s: float = 120.0
    c_puct: float = 1.4
    topk_templates: int = 20
    max_depth: int = 6
    max_children_per_node: int = 50
    llm_intervention_interval: int = 50
    use_filter: bool = True
    filter_cutoff: float = 0.0
    use_ringbreaker: bool = True
    ringbreaker_topk: int = 10
    max_routes: int = 3


@dataclass
class SearchResult:
    solved: bool
    first_solution_iteration: int | None
    search_time_s: float
    iterations: int
    root_children: int
    config_snapshot: dict
    routes: list[dict] = field(default_factory=list)


class SearchTree:
    """Minimal MCTS planner."""

    def __init__(
        self,
        target: Molecule,
        stock: InchiKeyStock,
        policy: OnnxTemplatePolicy,
        templates: TemplateLibrary,
        reaction_engine: ReactionEngine,
        config: SearchConfig,
        llm_controller: LLMMetaController | None = None,
        filter_policy: OnnxFilterPolicy | None = None,
        ringbreaker_policy: OnnxTemplatePolicy | None = None,
        ringbreaker_templates: TemplateLibrary | None = None,
    ):
        self.stock = stock
        self.policy = policy
        self.templates = templates
        self.reaction_engine = reaction_engine
        self.config = config
        self.llm_controller = llm_controller
        self.filter_policy = filter_policy
        self.ringbreaker_policy = ringbreaker_policy
        self.ringbreaker_templates = ringbreaker_templates
        self.root = SearchNode(state=SearchState(molecules=(target,), depth=0), prior=1.0)

        self._seen_states: set[tuple[str, ...]] = {self.root.state.key()}
        self._all_nodes: list[SearchNode] = [self.root]
        self._filter_runtime_error: str | None = None
        self.first_solution_iteration: int | None = None
        self.iterations: int = 0

    def run(self) -> SearchResult:
        t0 = time.time()

        for i in range(1, self.config.iteration_limit + 1):
            self.iterations = i
            if time.time() - t0 > self.config.time_limit_s:
                break

            node = self._select(self.root)
            if not node.expanded:
                self._expand(node)

            value = self._evaluate(node)
            self._backpropagate(node, value)

            if self.first_solution_iteration is None and node.state.is_solved(self.stock):
                self.first_solution_iteration = i

            if (
                self.llm_controller is not None
                and i % self.config.llm_intervention_interval == 0
            ):
                self.llm_controller.maybe_adjust(self)

        solved = self.first_solution_iteration is not None or self.root.state.is_solved(self.stock)
        return SearchResult(
            solved=solved,
            first_solution_iteration=self.first_solution_iteration,
            search_time_s=time.time() - t0,
            iterations=self.iterations,
            root_children=len(self.root.children),
            config_snapshot=asdict(self.config),
            routes=self._collect_routes(self.config.max_routes),
        )

    def diagnostics(self) -> dict:
        return {
            "iterations": self.iterations,
            "root_children": len(self.root.children),
            "first_solution_iteration": self.first_solution_iteration,
            "c_puct": self.config.c_puct,
            "topk_templates": self.config.topk_templates,
            "use_filter": self.config.use_filter and self.filter_policy is not None,
            "use_ringbreaker": self.config.use_ringbreaker
            and self.ringbreaker_policy is not None
            and self.ringbreaker_templates is not None,
            "filter_runtime_error": self._filter_runtime_error,
        }

    def _select(self, node: SearchNode) -> SearchNode:
        current = node
        while current.expanded and current.children:
            current = current.best_child(self.config.c_puct)
        return current

    def _expand(self, node: SearchNode) -> None:
        if node.state.depth >= self.config.max_depth:
            node.expanded = True
            return

        expandable = node.state.expandable(self.stock)
        if not expandable:
            node.expanded = True
            return

        target = expandable[0]
        children_added = self._expand_with_policy(
            node=node,
            target=target,
            policy=self.policy,
            templates=self.templates,
            topk=self.config.topk_templates,
            policy_source="uspto",
        )
        if (
            children_added == 0
            and self.config.use_ringbreaker
            and self.ringbreaker_policy is not None
            and self.ringbreaker_templates is not None
        ):
            self._expand_with_policy(
                node=node,
                target=target,
                policy=self.ringbreaker_policy,
                templates=self.ringbreaker_templates,
                topk=self.config.ringbreaker_topk,
                policy_source="ringbreaker",
            )
        node.expanded = True

    def _expand_with_policy(
        self,
        node: SearchNode,
        target: Molecule,
        policy: OnnxTemplatePolicy,
        templates: TemplateLibrary,
        topk: int,
        policy_source: str,
    ) -> int:
        fp = target.morgan_fingerprint(radius=2, nbits=2048)
        predictions = policy.predict_topk(fp, topk=topk)

        children_added = 0
        for pred in predictions:
            try:
                smarts = templates.smarts_by_index(pred.index)
            except Exception:
                continue
            outcomes = self.reaction_engine.apply(target, smarts)
            for outcome in outcomes:
                filter_score: float | None = None
                if self.filter_policy is not None and self.config.use_filter:
                    try:
                        filter_score = self.filter_policy.score_reaction(
                            target, outcome.reactants
                        )
                    except Exception as exc:
                        self._filter_runtime_error = str(exc)
                        self.filter_policy = None
                        filter_score = None
                    if (
                        filter_score is not None
                        and filter_score < self.config.filter_cutoff
                    ):
                        continue

                new_molecules = list(node.state.molecules)
                new_molecules.remove(target)
                new_molecules.extend(outcome.reactants)
                child_state = SearchState(
                    molecules=tuple(new_molecules), depth=node.state.depth + 1
                )
                key = child_state.key()
                if key in self._seen_states:
                    continue

                self._seen_states.add(key)
                child = SearchNode(
                    state=child_state,
                    prior=max(pred.probability, 1e-8),
                    parent=node,
                    transition=NodeTransition(
                        expanded_molecule=target.smiles,
                        reactants=tuple(m.smiles for m in outcome.reactants),
                        template_index=pred.index,
                        template_smarts=smarts,
                        policy_source=policy_source,
                        policy_probability=pred.probability,
                        filter_score=filter_score,
                    ),
                )
                node.children.append(child)
                self._all_nodes.append(child)
                children_added += 1
                if len(node.children) >= self.config.max_children_per_node:
                    return children_added
        return children_added

    def _evaluate(self, node: SearchNode) -> float:
        mask = node.state.in_stock_mask(self.stock)
        solved_ratio = sum(mask) / max(1, len(mask))
        if all(mask):
            return 1.0
        depth_penalty = min(node.state.depth / max(1, self.config.max_depth), 1.0) * 0.2
        return max(0.0, solved_ratio - depth_penalty)

    def _backpropagate(self, node: SearchNode, value: float) -> None:
        current: SearchNode | None = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent

    def _collect_routes(self, max_routes: int) -> list[dict]:
        if max_routes <= 0:
            return []

        candidates = [node for node in self._all_nodes if node is not self.root]
        if not candidates:
            return []

        solved = [node for node in candidates if node.state.is_solved(self.stock)]
        pool = solved or candidates
        ranked = sorted(
            pool,
            key=lambda node: (node.q_value(), node.visits, -node.state.depth),
            reverse=True,
        )

        routes: list[dict] = []
        seen_signatures: set[tuple[tuple[int, tuple[str, ...]], ...]] = set()
        for node in ranked:
            route = self._extract_route(node)
            signature = tuple(
                (int(step["template_index"]), tuple(step["reactants"]))
                for step in route["steps"]
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            routes.append(route)
            if len(routes) >= max_routes:
                break
        return routes

    def _extract_route(self, node: SearchNode) -> dict:
        transitions: list[NodeTransition] = []
        cursor: SearchNode | None = node
        while cursor is not None and cursor.parent is not None:
            if cursor.transition is None:
                break
            transitions.append(cursor.transition)
            cursor = cursor.parent
        transitions.reverse()

        steps: list[dict] = []
        for idx, transition in enumerate(transitions, start=1):
            steps.append(
                {
                    "step": idx,
                    "expanded_molecule": transition.expanded_molecule,
                    "reactants": list(transition.reactants),
                    "template_index": transition.template_index,
                    "template_smarts": transition.template_smarts,
                    "policy_source": transition.policy_source,
                    "policy_probability": transition.policy_probability,
                    "filter_score": transition.filter_score,
                }
            )

        return {
            "score": node.q_value(),
            "visits": node.visits,
            "depth": node.state.depth,
            "solved": node.state.is_solved(self.stock),
            "molecules": [mol.smiles for mol in node.state.molecules],
            "steps": steps,
        }
