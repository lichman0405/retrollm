"""Public artifact downloader.

This is a standalone implementation that downloads the same public artifacts as
AiZynthFinder, and writes a compatible `config.yml` into the target folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


@dataclass(frozen=True)
class PublicFile:
    key: str
    filename: str
    url: str


PUBLIC_FILES = [
    PublicFile(
        key="policy_model_onnx",
        filename="uspto_model.onnx",
        url="https://zenodo.org/record/7797465/files/uspto_model.onnx",
    ),
    PublicFile(
        key="template_file",
        filename="uspto_templates.csv.gz",
        url="https://zenodo.org/record/7341155/files/uspto_unique_templates.csv.gz",
    ),
    PublicFile(
        key="ringbreaker_model_onnx",
        filename="uspto_ringbreaker_model.onnx",
        url="https://zenodo.org/record/7797465/files/uspto_ringbreaker_model.onnx",
    ),
    PublicFile(
        key="ringbreaker_templates",
        filename="uspto_ringbreaker_templates.csv.gz",
        url="https://zenodo.org/record/7341155/files/uspto_ringbreaker_unique_templates.csv.gz",
    ),
    PublicFile(
        key="stock",
        filename="zinc_stock.hdf5",
        url="https://ndownloader.figshare.com/files/23086469",
    ),
    PublicFile(
        key="filter_policy_onnx",
        filename="uspto_filter_model.onnx",
        url="https://zenodo.org/record/7797465/files/uspto_filter_model.onnx",
    ),
]


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name)
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()


def write_artifacts_config(target_dir: Path) -> Path:
    """Write a config.yml compatible with the AiZynthFinder public data layout."""
    target_dir = target_dir.resolve()
    mapping = {pf.key: str((target_dir / pf.filename).as_posix()) for pf in PUBLIC_FILES}

    config_text = (
        "expansion:\n"
        "  uspto:\n"
        f"    - {mapping['policy_model_onnx']}\n"
        f"    - {mapping['template_file']}\n"
        "  ringbreaker:\n"
        f"    - {mapping['ringbreaker_model_onnx']}\n"
        f"    - {mapping['ringbreaker_templates']}\n"
        "filter:\n"
        f"  uspto: {mapping['filter_policy_onnx']}\n"
        "stock:\n"
        f"  zinc: {mapping['stock']}\n"
    )
    out_path = target_dir / "config.yml"
    out_path.write_text(config_text, encoding="utf-8")
    return out_path


def download_public_data(target_dir: str | Path) -> Path:
    """Download public artifacts into a directory and write config.yml."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for pf in PUBLIC_FILES:
        _download(pf.url, target_dir / pf.filename)
    return write_artifacts_config(target_dir)
