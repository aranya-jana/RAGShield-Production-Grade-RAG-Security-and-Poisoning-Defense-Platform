import os

from src.api import info


def test_info_reports_deployment_mode(monkeypatch):
    monkeypatch.setenv(
        "RAGSHIELD_DEPLOYMENT_MODE",
        "lite",
    )

    result = info()

    assert result["deployment_mode"] == "lite"


def test_info_defaults_to_full_deployment_mode(monkeypatch):
    monkeypatch.delenv(
        "RAGSHIELD_DEPLOYMENT_MODE",
        raising=False,
    )

    result = info()

    assert result["deployment_mode"] == "full"
