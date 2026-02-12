from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from retrollm.chem import Molecule
from retrollm.search.node import SearchNode
from retrollm.search.state import SearchState


class _DummyStock:
    def __contains__(self, mol: Molecule) -> bool:
        return mol.smiles == "CCO"


def test_state_expandable_and_solved() -> None:
    stock = _DummyStock()
    s1 = SearchState((Molecule("CCO"),), depth=0)
    s2 = SearchState((Molecule("CCO"), Molecule("CCN")), depth=0)

    assert s1.is_solved(stock) is True
    assert s2.is_solved(stock) is False
    assert len(s2.expandable(stock)) == 1


def test_node_ucb_score_runs() -> None:
    root = SearchNode(state=SearchState((Molecule("CCN"),), depth=0))
    child = SearchNode(state=SearchState((Molecule("CCO"),), depth=1), parent=root, prior=0.2)
    root.children.append(child)
    root.visits = 10
    child.visits = 2
    child.value_sum = 1.0

    score = child.ucb_score(c_puct=1.4)
    assert isinstance(score, float)
