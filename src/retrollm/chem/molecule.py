"""Molecule utilities.

This is a minimal wrapper around RDKit that exposes:
- canonical SMILES
- InChIKey
- Morgan fingerprint compatible with the ONNX policy input
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


@dataclass(frozen=True, slots=True)
class Molecule:
    smiles: str
    _rdmol: Chem.Mol = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {self.smiles}")
        object.__setattr__(self, "_rdmol", mol)
        can = Chem.MolToSmiles(mol)
        object.__setattr__(self, "smiles", can)

    @property
    def rdmol(self) -> Chem.Mol:
        return self._rdmol

    @property
    def inchikey(self) -> str:
        return Chem.MolToInchiKey(self.rdmol)

    def morgan_fingerprint(self, radius: int = 2, nbits: int = 2048) -> np.ndarray:
        fp = AllChem.GetMorganFingerprintAsBitVect(self.rdmol, radius, nbits)
        arr = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
