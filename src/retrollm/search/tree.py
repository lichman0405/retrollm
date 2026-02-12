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
    constraints_text: str = ""
    llm_use_subgoal_advisor: bool = True
    llm_use_route_reranker: bool = True
    llm_use_failure_diagnosis: bool = True
    llm_use_handoff: bool = True
    llm_retry_on_failure: bool = True
    llm_max_retry_attempts: int = 1


@dataclass
class SearchResult:
    solved: bool
    first_solution_iteration: int | None
    search_time_s: float
    iterations: int
    root_children: int
    config_snapshot: dict
    routes: list[dict] = field(default_factory=list)
    retry_attempts: int = 0
    llm: dict = field(default_factory=dict)


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
        constraints: dict | None = None,
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
        self.constraints = constraints or {}
        self.root = SearchNode(state=SearchState(molecules=(target,), depth=0), prior=1.0)

        self._seen_states: set[tuple[str, ...]] = {self.root.state.key()}
        self._all_nodes: list[SearchNode] = [self.root]
        self._filter_runtime_error: str | None = None
        self._constraint_drop_count = 0
        self._subgoal_advisor_calls = 0
        self._rerank_meta: dict = {"applied": False, "reason": "disabled_or_unavailable"}
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

        routes = self._collect_routes(self.config.max_routes * 3)
        if (
            self.llm_controller is not None
            and self.config.llm_use_route_reranker
            and routes
            and hasattr(self.llm_controller, "rerank_routes")
        ):
            try:
                routes, self._rerank_meta = self.llm_controller.rerank_routes(
                    routes,
                    objective={
                        "constraints": self.constraints,
                        "first_solution_iteration": self.first_solution_iteration,
                        "iterations": self.iterations,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self._rerank_meta = {
                    "applied": False,
                    "reason": "rerank_error",
                    "error": str(exc),
                }
        routes = routes[: self.config.max_routes]

        solved = self.first_solution_iteration is not None or self.root.state.is_solved(self.stock)
        llm_payload = {
            "constraints": self.constraints,
            "filter_runtime_error": self._filter_runtime_error,
            "constraint_drop_count": self._constraint_drop_count,
            "subgoal_advisor_calls": self._subgoal_advisor_calls,
            "rerank": self._rerank_meta,
            "final_diagnostics": self.diagnostics(),
            "events": self.llm_controller.events_as_dict()
            if self.llm_controller is not None
            else [],
        }
        return SearchResult(
            solved=solved,
            first_solution_iteration=self.first_solution_iteration,
            search_time_s=time.time() - t0,
            iterations=self.iterations,
            root_children=len(self.root.children),
            config_snapshot=asdict(self.config),
            routes=routes,
            llm=llm_payload,
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
            "constraint_drop_count": self._constraint_drop_count,
            "subgoal_advisor_calls": self._subgoal_advisor_calls,
        }

    def _select(self, node: SearchNode) -> SearchNode:
        current = node
        while current.expanded and current.children:
            current = current.best_child(self.config.c_puct)
        return current

    def _expand(self, node: SearchNode) -> None:
        max_steps_constraint = self._constraint_max_steps()
        effective_max_depth = self.config.max_depth
        if max_steps_constraint is not None:
            effective_max_depth = min(effective_max_depth, max_steps_constraint)

        if node.state.depth >= effective_max_depth:
            node.expanded = True
            return

        expandable = node.state.expandable(self.stock)
        if not expandable:
            node.expanded = True
            return

        target = self._select_expandable_target(node, expandable)
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
                next_depth = node.state.depth + 1
                if self._should_drop_by_constraints(
                    reactant_smiles=[m.smiles for m in outcome.reactants],
                    template_index=pred.index,
                    next_depth=next_depth,
                ):
                    self._constraint_drop_count += 1
                    continue

                new_molecules = list(node.state.molecules)
                new_molecules.remove(target)
                new_molecules.extend(outcome.reactants)
                child_state = SearchState(
                    molecules=tuple(new_molecules), depth=next_depth
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

    def _constraint_max_steps(self) -> int | None:
        raw = self.constraints.get("max_steps")
        if raw is None:
            return None
        try:
            value = int(raw)
            return value if value > 0 else None
        except Exception:
            return None

    def _select_expandable_target(
        self, node: SearchNode, expandable: tuple[Molecule, ...]
    ) -> Molecule:
        if len(expandable) == 1:
            return expandable[0]

        if self.constraints.get("prefer_in_stock_subgoals"):
            # Expand smaller molecules first as a weak proxy for purchasability.
            return min(expandable, key=lambda mol: len(mol.smiles))

        if (
            self.llm_controller is not None
            and self.config.llm_use_subgoal_advisor
            and hasattr(self.llm_controller, "choose_expansion_target")
        ):
            try:
                choice = self.llm_controller.choose_expansion_target(
                    expandable_smiles=[mol.smiles for mol in expandable],
                    context={
                        "depth": node.state.depth,
                        "num_expandable": len(expandable),
                        "diagnostics": self.diagnostics(),
                        "constraints": self.constraints,
                    },
                )
                idx = int(choice.get("index", 0))
                idx = max(0, min(len(expandable) - 1, idx))
                self._subgoal_advisor_calls += 1
                return expandable[idx]
            except Exception:
                pass
        return expandable[0]

    def _should_drop_by_constraints(
        self, reactant_smiles: list[str], template_index: int, next_depth: int
    ) -> bool:
        max_steps = self._constraint_max_steps()
        if max_steps is not None and next_depth > max_steps:
            return True

        avoid_templates = self.constraints.get("avoid_template_indices", [])
        if isinstance(avoid_templates, list):
            try:
                avoid_set = {int(item) for item in avoid_templates}
            except Exception:
                avoid_set = set()
            if template_index in avoid_set:
                return True

        avoid_reactants = self.constraints.get("avoid_reactants", [])
        if isinstance(avoid_reactants, list):
            lowered = [smiles.lower() for smiles in reactant_smiles]
            for token in avoid_reactants:
                token_s = str(token).strip().lower()
                if not token_s:
                    continue
                if any(token_s in smi for smi in lowered):
                    return True
        return False

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
