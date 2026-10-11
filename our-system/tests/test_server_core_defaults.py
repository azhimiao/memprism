"""The public launcher requires atomic spans; research requires explicit intent."""
import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    import os
    for name in list(os.environ):
        if name.startswith("AML_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AML_DB_PATH", str(tmp_path / "memory.sqlite"))
    spec = importlib.util.spec_from_file_location("aml_server_launcher", Path(__file__).parents[1] / "scripts/run_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_launcher_has_atomic_span_routes_and_no_optional_effects(launcher):
    settings = launcher.service_settings()
    assert settings.profile_id == "span_sync"
    assert settings.required_routes == ("raw", "span", "fts")
    assert settings.route_policy == "none"
    assert not settings.source_labels and not settings.claim_projection
    assert settings.unknown_dense_floor is None
    application = launcher.create_app(settings)
    with TestClient(application) as client:
        response = client.get("/health")
        assert response.status_code == 200


@pytest.mark.parametrize("profile", ["research", "raw_sync", "span_sync"])
def test_explicit_profile_is_preserved(launcher, monkeypatch, profile):
    monkeypatch.setenv("AML_PROFILE", profile)
    assert launcher.service_settings().profile_id == profile


@pytest.mark.parametrize("mode", ["raw_dense", "raw_hybrid_rrf"])
def test_dense_configuration_requires_all_span_routes(launcher, monkeypatch, mode):
    monkeypatch.setenv("AML_RETRIEVAL_MODE", mode)
    monkeypatch.setenv("AML_EMBEDDING_PROVIDER", "review-only")
    monkeypatch.setenv("AML_EMBEDDING_ENDPOINT", "https://example.invalid/embeddings")
    monkeypatch.setenv("AML_EMBEDDING_DIMENSIONS", "2")
    settings = launcher.service_settings()
    assert settings.profile_id == "span_dense_sync"
    assert settings.required_routes == ("raw", "span", "fts", "dense")
    assert settings.route_policy == "none"
    assert settings.route_mode == mode


def test_launcher_rejects_invalid_route(launcher, monkeypatch):
    monkeypatch.setenv("AML_ROUTE_POLICY", "oracle")
    with pytest.raises(ValueError, match="unknown optional route policy"):
        launcher.service_settings()
