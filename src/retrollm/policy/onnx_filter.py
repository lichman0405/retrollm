"""ONNX-based reaction filter policy.

The public USPTO filter model scores (product, reactants) reaction candidates.

Supported ONNX signatures:
- single input tensor (concatenated product/reactant vectors)
- two input tensors (product vector + reactant vector)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort


class OnnxFilterPolicy:
    """Scores reaction candidates with an ONNX model."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError("Filter model has no inputs")

        self._mode = "single" if len(inputs) == 1 else "pair"
        self._single_input_name: str | None = None
        self._single_input_dim: int | None = None
        self._product_input_name: str | None = None
        self._reactant_input_name: str | None = None
        self._product_dim: int = 2048
        self._reactant_dim: int = 2048

        if self._mode == "single":
            self._single_input_name = inputs[0].name
            self._single_input_dim = self._resolve_input_dim(inputs[0].shape, default=4096)
        else:
            self._product_input_name = inputs[0].name
            self._reactant_input_name = inputs[1].name
            self._product_dim = self._resolve_input_dim(inputs[0].shape, default=2048)
            self._reactant_dim = self._resolve_input_dim(inputs[1].shape, default=2048)

    def _resolve_input_dim(self, shape: list[int | str | None], default: int) -> int:
        if len(shape) >= 2 and isinstance(shape[-1], int):
            dim = int(shape[-1])
            if dim > 0:
                return dim
        return default

    def _merged_reactant_fp(
        self, reactants: tuple["Molecule", ...], nbits: int
    ) -> np.ndarray:
        merged = np.zeros((nbits,), dtype=np.float32)
        for mol in reactants:
            merged = np.maximum(merged, mol.morgan_fingerprint(nbits=nbits))
        return merged

    def _fit_dim(self, x: np.ndarray, dim: int) -> np.ndarray:
        if x.size == dim:
            return x
        if x.size < dim:
            out = np.zeros((dim,), dtype=np.float32)
            out[: x.size] = x
            return out
        return x[:dim]

    def _single_input_vector(
        self, product: "Molecule", reactants: tuple["Molecule", ...]
    ) -> np.ndarray:
        product_fp = product.morgan_fingerprint(nbits=2048)
        reactants_fp = self._merged_reactant_fp(reactants, nbits=2048)
        x = np.concatenate([product_fp, reactants_fp]).astype(np.float32, copy=False)
        dim = self._single_input_dim if self._single_input_dim is not None else 4096
        return self._fit_dim(x, dim)

    def _two_input_vectors(
        self, product: "Molecule", reactants: tuple["Molecule", ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        product_fp = product.morgan_fingerprint(nbits=self._product_dim)
        reactants_fp = self._merged_reactant_fp(reactants, nbits=self._reactant_dim)
        return self._fit_dim(product_fp, self._product_dim), self._fit_dim(
            reactants_fp, self._reactant_dim
        )

    def score_reaction(
        self, product: "Molecule", reactants: tuple["Molecule", ...]
    ) -> float:
        if self._mode == "single":
            if self._single_input_name is None:
                raise RuntimeError("Filter single-input model is not initialized")
            x = self._single_input_vector(product, reactants).reshape(1, -1)
            feed = {self._single_input_name: x}
        else:
            if self._product_input_name is None or self._reactant_input_name is None:
                raise RuntimeError("Filter two-input model is not initialized")
            product_x, reactant_x = self._two_input_vectors(product, reactants)
            feed = {
                self._product_input_name: product_x.reshape(1, -1),
                self._reactant_input_name: reactant_x.reshape(1, -1),
            }

        outputs = self.session.run(None, feed)
        raw = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if raw.size == 0:
            return 0.0
        score = float(raw[-1])
        if score < 0.0 or score > 1.0:
            score = 1.0 / (1.0 + math.exp(-score))
        return score


if TYPE_CHECKING:
    from retrollm.chem import Molecule
