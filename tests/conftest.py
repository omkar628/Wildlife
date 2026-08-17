"""Shared fixtures. Tests never require the full camera-trap dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.config import reload_settings
from backend.database.connection import Database, reset_database_singleton
from backend.api import deps


@pytest.fixture
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "wildlife.db"
    crops = tmp_path / "crops"
    logs = tmp_path / "logs"
    classified = tmp_path / "classified"
    monkeypatch.setenv("WI_DATABASE_PATH", str(db_path))
    monkeypatch.setenv("WI_CROPS_DIR", str(crops))
    monkeypatch.setenv("WI_CLASSIFIED_DIR", str(classified))
    monkeypatch.setenv("WI_LOGS_DIR", str(logs))
    monkeypatch.setenv("WI_REID_ENABLED", "false")
    reload_settings()
    reset_database_singleton()
    deps.get_detector.cache_clear()
    deps.get_pipeline.cache_clear()
    deps.get_review_service.cache_clear()
    deps.get_graph_service.cache_clear()
    deps.get_reid_adapter.cache_clear()
    deps.get_reid_encoder.cache_clear()
    deps.get_gnn_service.cache_clear()
    deps.get_identity_service.cache_clear()
    deps.get_alert_service.cache_clear()
    settings = reload_settings()
    yield settings
    reload_settings()
    reset_database_singleton()


@pytest.fixture
def db(tmp_settings) -> Database:
    from backend.database.connection import get_database

    return get_database(tmp_settings.database_path)
