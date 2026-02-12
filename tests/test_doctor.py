from __future__ import annotations

from pathlib import Path

from retrollm import doctor


def test_check_llm_env_warn_when_provider_missing(monkeypatch) -> None:
    monkeypatch.delenv("RETROLLM_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RETROLLM_LLM_MODEL", raising=False)
    check = doctor._check_llm_env()
    assert check.status == "warn"


def test_check_llm_env_fail_for_openai_missing_fields(monkeypatch) -> None:
    monkeypatch.setenv("RETROLLM_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("RETROLLM_LLM_MODEL", "gpt-test")
    monkeypatch.delenv("RETROLLM_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("RETROLLM_LLM_API_KEY", raising=False)
    check = doctor._check_llm_env()
    assert check.status == "fail"


def test_run_doctor_missing_config_reports_failure(tmp_path: Path) -> None:
    report = doctor.run_doctor(tmp_path / "missing_config.yml")
    assert any(
        check.name == "Artifacts config" and check.status == "fail"
        for check in report.checks
    )
