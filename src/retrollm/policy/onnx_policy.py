"""ONNX-based template policy.

The public USPTO model expects a Morgan fingerprint vector as input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class TemplatePrediction:
    index: int
    probability: float


class OnnxTemplatePolicy:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name

    def predict_topk(
        self, fingerprint: np.ndarray, topk: int = 50
    ) -> list[TemplatePrediction]:
        if fingerprint.ndim != 1:
            raise ValueError("fingerprint must be a 1D vector")
        x = fingerprint.astype(np.float32, copy=False).reshape(1, -1)
        outputs = self.session.run(None, {self._input_name: x})
        probs = np.asarray(outputs[0]).reshape(-1)
        if topk <= 0:
            return []
        topk = min(topk, probs.shape[0])
        idx = np.argpartition(probs, -topk)[-topk:]
        idx = idx[np.argsort(probs[idx])[::-1]]
        return [TemplatePrediction(int(i), float(probs[i])) for i in idx]
