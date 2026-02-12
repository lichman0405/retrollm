"""Search state representation."""

from __future__ import annotations

from dataclasses import dataclass

from retrollm.chem import Molecule
from retrollm.stock import InchiKeyStock


@dataclass(frozen=True)
class SearchState:
    """A state contains a tuple of molecules and a depth."""

    molecules: tuple[Molecule, ...]
    depth: int

    def in_stock_mask(self, stock: InchiKeyStock) -> tuple[bool, ...]:
        return tuple(m in stock for m in self.molecules)

    def is_solved(self, stock: InchiKeyStock) -> bool:
        return all(self.in_stock_mask(stock))

    def expandable(self, stock: InchiKeyStock) -> tuple[Molecule, ...]:
        mask = self.in_stock_mask(stock)
        return tuple(m for m, in_stock in zip(self.molecules, mask) if not in_stock)

    def key(self) -> tuple[str, ...]:
        return tuple(sorted(m.inchikey for m in self.molecules))
