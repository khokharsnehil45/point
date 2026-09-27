"""Global pytest fixtures and isolation for Point tests."""

from __future__ import annotations

from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def isolate_config(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests run isolated from the user's local config and environment."""
    temp_dir = tmp_path_factory.mktemp("point_test_config")
    monkeypatch.setattr("point.config.CONFIG_DIR", temp_dir)
    monkeypatch.setattr("point.config.CONFIG_FILE", temp_dir / "config.json")
    monkeypatch.delenv("POINT_MODEL", raising=False)
