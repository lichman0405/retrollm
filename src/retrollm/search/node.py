"""MCTS node."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from retrollm.search.state import SearchState


@dataclass(frozen=True)
class NodeTransition:
    """How this node was created from its parent."""

    expanded_molecule: str
    reactants: tuple[str, ...]
    template_index: int
    template_smarts: str
    policy_source: str
    policy_probability: float
    filter_score: float | None = None


@dataclass
class SearchNode:
    """Node in MCTS tree."""

    state: SearchState
    prior: float = 1.0
    parent: "SearchNode | None" = None
    transition: NodeTransition | None = None
    children: list["SearchNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False

    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb_score(self, c_puct: float) -> float:
        if self.parent is None:
            return self.q_value()
        prior_term = c_puct * self.prior * math.sqrt(max(1, self.parent.visits)) / (1 + self.visits)
        return self.q_value() + prior_term

    def best_child(self, c_puct: float) -> "SearchNode":
        return max(self.children, key=lambda node: node.ucb_score(c_puct))
