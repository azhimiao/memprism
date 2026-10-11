from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings


@pytest.fixture
def api(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    app = create_app(Settings(db_path=db_path))
    with TestClient(app) as client:
        yield client, db_path
