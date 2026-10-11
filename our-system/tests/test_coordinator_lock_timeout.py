"""Coordinator SQLite lock wait configuration and contention behavior."""
from __future__ import annotations

import math
import os
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from test_k0 import Provider, body

MARKER = "coordinator-lock-timeout-marker"


def _clear_aml_env(monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith("AML_"):
            monkeypatch.delenv(name, raising=False)


def test_default_coordinator_lock_timeout_is_point_one(tmp_path) -> None:
    settings = Settings(db_path=tmp_path / "defaults.db")
    settings.ensure_phase_a_defaults()
    assert settings.coordinator_lock_timeout_seconds == 0.1


def test_coordinator_lock_timeout_from_env(monkeypatch, tmp_path) -> None:
    _clear_aml_env(monkeypatch)
    monkeypatch.setenv("AML_DB_PATH", str(tmp_path / "env.db"))
    monkeypatch.setenv("AML_PROFILE", "raw_sync")
    monkeypatch.setenv("AML_COORDINATOR_LOCK_TIMEOUT_SECONDS", "2.5")
    settings = Settings.from_env()
    settings.ensure_phase_a_defaults()
    assert settings.coordinator_lock_timeout_seconds == 2.5


@pytest.mark.parametrize(
    "value",
    [0, -1, 31, math.nan, math.inf],
)
def test_invalid_coordinator_lock_timeout_rejected(tmp_path, value) -> None:
    settings = Settings(db_path=tmp_path / "bad.db", coordinator_lock_timeout_seconds=value)
    with pytest.raises(ValueError, match="coordinator lock timeout"):
        settings.ensure_phase_a_defaults()


def test_bool_coordinator_lock_timeout_rejected(tmp_path) -> None:
    settings = Settings(db_path=tmp_path / "bool.db", coordinator_lock_timeout_seconds=True)
    with pytest.raises(ValueError, match="coordinator lock timeout"):
        settings.ensure_phase_a_defaults()


def _hold_write_lock(db_path, ready: threading.Event, hold_seconds: float = 0.6) -> None:
    connection = sqlite3.connect(db_path, timeout=30)
    connection.execute("BEGIN IMMEDIATE")
    ready.set()
    time.sleep(hold_seconds)
    connection.execute("COMMIT")
    connection.close()


def _add_while_lock_held(tmp_path, lock_timeout: float):
    db_path = tmp_path / f"contention-{lock_timeout}.db"
    settings = Settings(
        db_path=db_path,
        profile_id="cycle2_dense",
        coordinator_lock_timeout_seconds=lock_timeout,
    )
    application = create_app(settings, embedding_provider=Provider())
    lock_ready = threading.Event()
    holder = threading.Thread(
        target=_hold_write_lock,
        args=(db_path, lock_ready),
        daemon=True,
    )
    holder.start()
    assert lock_ready.wait(timeout=5), "background lock was not acquired"
    with TestClient(application) as client:
        response = client.post("/add", json=body("lock-req", (MARKER,), user="lock-user"))
        return client, response, holder


def test_short_lock_timeout_returns_control_not_ready(tmp_path) -> None:
    client, response, holder = _add_while_lock_held(tmp_path, 0.1)
    holder.join(timeout=5)
    assert response.status_code == 503, response.json()
    assert response.json()["detail"] == "control_not_ready"
    assert response.headers.get("retry-after") == "1"
    found = client.post("/search", json={"user_id": "lock-user", "query": "marker", "top_k": 10})
    assert found.status_code == 200, found.json()
    assert not any(MARKER in item.get("content", "") for item in found.json().get("data", []))


def test_longer_lock_timeout_waits_and_add_is_searchable(tmp_path) -> None:
    client, response, holder = _add_while_lock_held(tmp_path, 2.0)
    holder.join(timeout=5)
    assert response.status_code == 200, response.json()
    found = client.post("/search", json={"user_id": "lock-user", "query": "marker", "top_k": 10})
    assert found.status_code == 200, found.json()
    payload = found.json()
    assert any(MARKER in item.get("content", "") for item in payload.get("data", []))
