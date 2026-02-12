from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
from retrollm.policy.onnx_policy import OnnxTemplatePolicy


def test_policy_predict_topk_shape() -> None:
    # Structural unit test. The model file must exist to run.
    model_path = Path("data/uspto_model.onnx")
    if not model_path.exists():
        return

    policy = OnnxTemplatePolicy(model_path)
    fp = np.zeros((2048,), dtype=np.float32)
    preds = policy.predict_topk(fp, topk=10)
    assert len(preds) == 10
