"""Stock query backend.

This first implementation uses an in-memory InChIKey set from HDF5/CSV/text.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retrollm.chem import Molecule


class InchiKeyStock:
    """Simple stock backend with in-memory key lookup."""

    def __init__(self, path: str | Path, inchi_key_col: str = "inchi_key"):
        self.path = Path(path)
        self.inchi_key_col = inchi_key_col
        self._keys = self._load_keys()

    def _load_keys(self) -> set[str]:
        ext = self.path.suffix.lower()
        if ext in {".h5", ".hdf5"}:
            df = pd.read_hdf(self.path, key="table")
            return set(df[self.inchi_key_col].astype(str).tolist())
        if ext == ".csv":
            df = pd.read_csv(self.path, usecols=[self.inchi_key_col])
            return set(df[self.inchi_key_col].astype(str).tolist())
        text = self.path.read_text(encoding="utf-8")
        return {line.strip() for line in text.splitlines() if line.strip()}

    def __contains__(self, mol: Molecule) -> bool:
        return mol.inchikey in self._keys

    def __len__(self) -> int:
        return len(self._keys)
