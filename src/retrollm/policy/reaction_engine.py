"""Template application engine based on RDChiral with RDKit fallback."""

from __future__ import annotations

from dataclasses import dataclass

from rdchiral.main import rdchiralReactants, rdchiralReaction, rdchiralRun
from rdkit import Chem
from rdkit.Chem import AllChem

from retrollm.chem import Molecule


@dataclass(frozen=True)
class ReactionOutcome:
    reactants: tuple[Molecule, ...]


class ReactionEngine:
    """Apply retrosynthesis templates to product molecules."""

    def apply(self, product: Molecule, smarts: str) -> list[ReactionOutcome]:
        outcomes = self._apply_rdchiral(product, smarts)
        if outcomes:
            return outcomes
        return self._apply_rdkit(product, smarts)

    def _apply_rdchiral(self, product: Molecule, smarts: str) -> list[ReactionOutcome]:
        try:
            rxn = rdchiralReaction(smarts)
            reactants = rdchiralReactants(product.smiles)
            results = rdchiralRun(rxn, reactants)
        except Exception:
            return []

        outcomes: list[ReactionOutcome] = []
        for joined in results:
            try:
                mols = tuple(Molecule(smi) for smi in joined.split("."))
            except Exception:
                continue
            outcomes.append(ReactionOutcome(reactants=mols))
        return outcomes

    def _apply_rdkit(self, product: Molecule, smarts: str) -> list[ReactionOutcome]:
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn is None:
                return []
            prod = Chem.MolFromSmiles(product.smiles)
            if prod is None:
                return []
            results = rxn.RunReactants([prod])
        except Exception:
            return []

        outcomes: list[ReactionOutcome] = []
        for reactant_set in results:
            smi_list: list[str] = []
            ok = True
            for mol in reactant_set:
                try:
                    smi = Chem.MolToSmiles(mol)
                    smi_list.append(smi)
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            try:
                outcome = tuple(Molecule(smi) for smi in smi_list)
            except Exception:
                continue
            outcomes.append(ReactionOutcome(reactants=outcome))
        return outcomes
