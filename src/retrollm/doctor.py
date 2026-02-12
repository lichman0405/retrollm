"""Environment and artifact diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    hint: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
        }


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [check.as_dict() for check in self.checks]}


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _check_python_version() -> CheckResult:
    major, minor = sys.version_info.major, sys.version_info.minor
    version = f"{major}.{minor}"
    if (major, minor) < (3, 10):
        return CheckResult(
            name="Python version",
            status="fail",
            detail=f"Found {version}, requires >=3.10,<3.13.",
            hint="Use Python 3.11 for best compatibility.",
        )
    if (major, minor) >= (3, 13):
        return CheckResult(
            name="Python version",
            status="fail",
            detail=f"Found {version}, requires >=3.10,<3.13.",
            hint="Create a conda env with python=3.11 and reinstall retrollm.",
        )
    return CheckResult(
        name="Python version",
        status="ok",
        detail=f"{version} is within supported range.",
    )


def _check_virtual_env() -> CheckResult:
    in_env = bool(
        os.environ.get("VIRTUAL_ENV")
        or os.environ.get("CONDA_PREFIX")
        or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    )
    if in_env:
        return CheckResult(
            name="Virtual environment",
            status="ok",
            detail="Detected active venv/conda environment.",
        )
    return CheckResult(
        name="Virtual environment",
        status="warn",
        detail="No active venv/conda environment detected.",
        hint="Run commands from a dedicated environment to avoid dependency conflicts.",
    )


def _check_dependency(module_name: str, label: str) -> CheckResult:
    if _module_available(module_name):
        return CheckResult(name=f"Dependency: {label}", status="ok", detail="Installed.")
    return CheckResult(
        name=f"Dependency: {label}",
        status="fail",
        detail=f"Missing Python module '{module_name}'.",
        hint=f"Install {label} in the active environment.",
    )


def _resolve_artifact_path(raw_value: Any, config_path: Path) -> Path | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    p = Path(raw_value.strip())
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def _check_artifacts(config_path: Path) -> list[CheckResult]:
    if not config_path.exists():
        return [
            CheckResult(
                name="Artifacts config",
                status="fail",
                detail=f"Missing config file: {config_path}",
                hint="Run `retrollm download-data ./data` or provide --config path.",
            )
        ]

    try:
        import yaml
    except Exception:
        return [
            CheckResult(
                name="Artifacts config",
                status="fail",
                detail="PyYAML is not installed, cannot parse config.yml.",
                hint="Install project dependencies and rerun doctor.",
            )
        ]

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            CheckResult(
                name="Artifacts config",
                status="fail",
                detail=f"Failed to parse config: {exc}",
                hint="Regenerate config with `retrollm download-data ./data`.",
            )
        ]

    required_paths = {
        "expansion.uspto.model": data.get("expansion", {}).get("uspto", [None, None])[0],
        "expansion.uspto.templates": data.get("expansion", {}).get("uspto", [None, None])[1],
        "expansion.ringbreaker.model": data.get("expansion", {}).get("ringbreaker", [None, None])[0],
        "expansion.ringbreaker.templates": data.get("expansion", {}).get("ringbreaker", [None, None])[1],
        "filter.uspto": data.get("filter", {}).get("uspto"),
        "stock.zinc": data.get("stock", {}).get("zinc"),
    }

    checks: list[CheckResult] = [
        CheckResult(
            name="Artifacts config",
            status="ok",
            detail=f"Loaded config file: {config_path}",
        )
    ]
    for label, raw_path in required_paths.items():
        path = _resolve_artifact_path(raw_path, config_path)
        if path is None:
            checks.append(
                CheckResult(
                    name=f"Artifact: {label}",
                    status="fail",
                    detail="Missing path in config.",
                    hint="Regenerate config or fix this field manually.",
                )
            )
            continue
        if not path.exists():
            checks.append(
                CheckResult(
                    name=f"Artifact: {label}",
                    status="fail",
                    detail=f"Missing file: {path}",
                    hint="Download artifacts again and verify file permissions.",
                )
            )
            continue
        size = path.stat().st_size
        if size <= 0:
            checks.append(
                CheckResult(
                    name=f"Artifact: {label}",
                    status="fail",
                    detail=f"Empty file: {path}",
                    hint="Re-download artifacts, the file may be corrupted.",
                )
            )
            continue
        checks.append(
            CheckResult(
                name=f"Artifact: {label}",
                status="ok",
                detail=f"Found {path.name} ({size} bytes).",
            )
        )
    return checks


def _check_llm_env() -> CheckResult:
    provider = os.environ.get("RETROLLM_LLM_PROVIDER", "").strip()
    if not provider:
        return CheckResult(
            name="LLM configuration",
            status="warn",
            detail="RETROLLM_LLM_PROVIDER is not set; LLM meta-control is disabled.",
            hint="Set RETROLLM_LLM_* variables if you want --use-llm.",
        )

    model = os.environ.get("RETROLLM_LLM_MODEL", "").strip()
    if not model:
        return CheckResult(
            name="LLM configuration",
            status="fail",
            detail="RETROLLM_LLM_MODEL is required when provider is configured.",
            hint="Set RETROLLM_LLM_MODEL in .env.",
        )

    base_url = os.environ.get("RETROLLM_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("RETROLLM_LLM_API_KEY", "").strip()
    normalized = provider.lower()
    known_openai_compatible = {
        "openai_compatible",
        "openai-compatible",
        "openai",
        "deepseek",
        "deep_seek",
        "openrouter",
        "siliconflow",
        "moonshot",
        "kimi",
        "qwen",
    }
    is_custom_class_path = ":" in provider
    inferred_openai_compatible = (normalized in known_openai_compatible) or (
        not is_custom_class_path and bool(base_url)
    )

    if not is_custom_class_path and normalized not in known_openai_compatible and not base_url:
        return CheckResult(
            name="LLM configuration",
            status="fail",
            detail=(
                f"Provider '{provider}' is not a known alias and BASE_URL is empty."
            ),
            hint=(
                "Set RETROLLM_LLM_PROVIDER=openai_compatible (or deepseek/openrouter) "
                "with BASE_URL/API_KEY, or use class path 'package.module:ClassName'."
            ),
        )

    if inferred_openai_compatible and (not base_url or not api_key):
        return CheckResult(
            name="LLM configuration",
            status="fail",
            detail="OpenAI-compatible mode requires BASE_URL and API_KEY.",
            hint="Set RETROLLM_LLM_BASE_URL and RETROLLM_LLM_API_KEY in .env.",
        )

    return CheckResult(
        name="LLM configuration",
        status="ok",
        detail=(
            f"Provider '{provider}' is configured"
            + (" (openai-compatible mode)." if inferred_openai_compatible else ".")
        ),
    )


def run_doctor(config_path: str | Path = "./data/config.yml") -> DoctorReport:
    path = Path(config_path).resolve()
    checks: list[CheckResult] = [
        _check_python_version(),
        _check_virtual_env(),
        _check_dependency("numpy", "NumPy"),
        _check_dependency("pandas", "Pandas"),
        _check_dependency("tables", "PyTables"),
        _check_dependency("onnxruntime", "ONNX Runtime"),
        _check_dependency("rdkit", "RDKit"),
        _check_dependency("rdchiral", "RDChiral"),
        _check_dependency("yaml", "PyYAML"),
        _check_dependency("rich", "Rich"),
        _check_llm_env(),
    ]
    checks.extend(_check_artifacts(path))
    return DoctorReport(checks=tuple(checks))
