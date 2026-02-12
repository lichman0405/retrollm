"""Configuration loader.

RetroLLM reuses the *artifact configuration* format produced by the public-data
downloader (compatible with AiZynthFinder's published artifacts).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class ArtifactsConfig:
    """Paths to local artifacts used by the pipeline."""

    policy_model_onnx: Path
    template_file: Path
    filter_policy_onnx: Path
    ringbreaker_model_onnx: Path
    ringbreaker_templates: Path
    stock: Path


def _require_path(value: Any, field: str) -> Path:
    if not value or not isinstance(value, str):
        raise ValueError(f"Missing or invalid config field: {field}")
    return Path(value)


def load_artifacts_config(path: str | Path) -> ArtifactsConfig:
    path = Path(path)
    data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    policy_model_onnx = _require_path(data["expansion"]["uspto"][0], "expansion.uspto[0]")
    template_file = _require_path(data["expansion"]["uspto"][1], "expansion.uspto[1]")
    ringbreaker_model_onnx = _require_path(
        data["expansion"]["ringbreaker"][0], "expansion.ringbreaker[0]"
    )
    ringbreaker_templates = _require_path(
        data["expansion"]["ringbreaker"][1], "expansion.ringbreaker[1]"
    )
    filter_policy_onnx = _require_path(data["filter"]["uspto"], "filter.uspto")
    stock = _require_path(data["stock"]["zinc"], "stock.zinc")

    return ArtifactsConfig(
        policy_model_onnx=policy_model_onnx,
        template_file=template_file,
        filter_policy_onnx=filter_policy_onnx,
        ringbreaker_model_onnx=ringbreaker_model_onnx,
        ringbreaker_templates=ringbreaker_templates,
        stock=stock,
    )
