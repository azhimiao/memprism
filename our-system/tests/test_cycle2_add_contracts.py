"""Cycle2 Add contracts for derived dispatch, same-batch retraction, and empty health.

Offline doubles only. No network and no paid provider calls.
"""
from __future__ import annotations

from dataclasses import replace
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import atomic_memory as am
from app.api import create_app
from app.config import Settings
from app.embeddings import prepare_embedding
from test_atomic_memory import StubProvider, draft
from test_k0 import Provider

STATEMENT = "I enjoy tea when fasting."
RETRACT = 'I retract my statement that "I enjoy tea when fasting."'
MARKER = "ATOMTEXTMARKER"
PUBLISHED = (
    "raw_messages", "source_spans", "atomic_memories", "atomic_vectors",
    "atomic_extractions", "add_requests", "k0_receipts", "k0_watermarks",
)


class LocalAtoms(StubProvider):
    def __init__(self, settings, empty=False):
        super().__init__(config=am.AtomicMemoryConfig.from_settings(settings))
        self.empty = empty
        self.seen = []

    def extract(self, messages, *, before_dispatch=None):
        if before_dispatch is not None:
            before_dispatch()
        self.seen = messages
        self.calls.append([message.id for message in messages])
        atoms = [] if self.empty else [draft(message, text=MARKER) for message in messages]
        return am.ExtractionBatch({"atoms": atoms})


def payload(texts, request="r", user="u"):
    return dict(user_id=user, session_id="s", request_id=request,
                messages=[dict(role="user", content=text) for text in texts])


def publication(path):
    with sqlite3.connect(path) as db:
        return {name: db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in PUBLISHED}


def assert_unpublished(path):
    observed = publication(path)
    assert all(value == 0 for value in observed.values()), observed


def build(tmp_path, profile, *, enabled=True, empty=False, name="add.db"):
    settings = Settings(db_path=tmp_path / name, profile_id=profile, enable_extractor=enabled,
                        atom_endpoint="https://mini.test/chat")
    atoms = LocalAtoms(settings, empty=empty) if enabled else None
    vectors = Provider() if profile == "cycle2_dense" else None
    application = create_app(settings, atom_provider=atoms, embedding_provider=vectors)
    return settings, atoms, vectors, application


def search(client, query, user="u"):
    return client.post("/search", json=dict(user_id=user, query=query, top_k=20))


def test_nonempty_dense_atom_add_search_recall(tmp_path):
    settings, atoms, vectors, application = build(tmp_path, "cycle2_dense")
    with TestClient(application) as client:
        added = client.post("/add", json=payload([STATEMENT]))
        assert added.status_code == 200, added.json()
        assert len(atoms.calls) == 1 and len(vectors.calls) == 2
        assert vectors.calls[0] == [STATEMENT] and MARKER in vectors.calls[1][0]
        found = search(client, "tea")
        assert found.status_code == 200, found.json()
        assert any(STATEMENT in item["content"] for item in found.json()["data"])
    observed = publication(settings.db_path)
    assert observed["atomic_memories"] > 0 and observed["atomic_vectors"] > 0
    with sqlite3.connect(settings.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM k0_attempts WHERE status='ready'").fetchone()[0] == 2


@pytest.mark.parametrize("enabled,empty", [(False, False), (True, True)])
def test_zero_atom_and_extractor_off_dense_add_unchanged(tmp_path, enabled, empty):
    settings, atoms, vectors, application = build(tmp_path, "cycle2_dense", enabled=enabled, empty=empty)
    with TestClient(application) as client:
        added = client.post("/add", json=payload([STATEMENT]))
        assert added.status_code == 200, added.json()
        assert vectors.calls == [[STATEMENT]]
        found = search(client, "tea")
        assert found.status_code == 200 and any(STATEMENT in item["content"] for item in found.json()["data"])
    observed = publication(settings.db_path)
    assert observed["atomic_vectors"] == 0
    assert observed["atomic_memories"] == 0
    assert atoms is None or atoms.calls == [[atoms.seen[0].id]]


def test_derived_permit_rejects_tampered_batch(tmp_path, monkeypatch):
    settings, atoms, vectors, application = build(tmp_path, "cycle2_dense", name="tamper.db")
    original = am.prepared_atom_texts

    def tamper(prepared):
        return [text + " TAMPER" for text in original(prepared)]

    monkeypatch.setattr(am, "prepared_atom_texts", tamper)
    with TestClient(application) as client:
        added = client.post("/add", json=payload([STATEMENT]))
        assert added.status_code == 503 and added.json()["detail"] == "dispatch_source_changed"
    assert_unpublished(settings.db_path)
    assert len(atoms.calls) == 1 and len(vectors.calls) == 1
    assert all("TAMPER" not in text and MARKER not in text for batch in vectors.calls for text in batch)


def test_derived_permit_rejects_extractor_identity_mismatch(tmp_path):
    settings, atoms, vectors, application = build(tmp_path, "cycle2_dense", name="identity.db")
    atoms.config = replace(atoms.config, revision="rejected")
    with TestClient(application) as client:
        added = client.post("/add", json=payload([STATEMENT]))
        assert added.status_code == 503 and added.json()["detail"] == "dispatch_source_changed"
    assert_unpublished(settings.db_path)
    assert len(vectors.calls) == 1
    assert all(MARKER not in text for batch in vectors.calls for text in batch)


def test_derived_permit_rejects_raw_dependency_changed_after_binding(tmp_path):
    settings, atoms, vectors, application = build(tmp_path, "cycle2_dense", name="dependency.db")

    def mutate():
        atoms.seen[0].content += " changed"

    vectors.action = mutate
    with TestClient(application) as client:
        added = client.post("/add", json=payload([STATEMENT]))
        assert added.status_code == 503 and added.json()["detail"] == "dispatch_source_changed"
    assert_unpublished(settings.db_path)
    assert len(vectors.calls) == 1 and vectors.calls[0] == [STATEMENT]
    assert all(MARKER not in text for batch in vectors.calls for text in batch)


def test_raw_permit_still_rejects_text_outside_spans(tmp_path, monkeypatch):
    settings, atoms, vectors, application = build(tmp_path, "cycle2_dense", name="raw-permit.db")
    import app.ingest as ingest

    def unbound(provider, texts):
        return prepare_embedding(provider, [*texts, "not-a-permitted-source"])

    monkeypatch.setattr(ingest, "prepare_embedding", unbound)
    with TestClient(application) as client:
        added = client.post("/add", json=payload([STATEMENT]))
        assert added.status_code == 503 and added.json()["detail"] == "dispatch_source_changed"
    assert_unpublished(settings.db_path)
    assert vectors.calls == []
    assert len(atoms.calls) == 1


@pytest.mark.parametrize("profile", ["cycle2_raw", "cycle2_dense"])
def test_same_batch_statement_then_retract_is_covered_not_disclosed(tmp_path, profile):
    settings, atoms, vectors, application = build(tmp_path, profile, name="retract.db")
    with TestClient(application) as client:
        added = client.post("/add", json=payload([STATEMENT, RETRACT]))
        assert added.status_code == 200, added.json()
        found = search(client, "tea")
        marker = search(client, MARKER)
        assert found.status_code == marker.status_code == 200
        assert found.json()["data"] == [] and marker.json()["data"] == []
    with sqlite3.connect(settings.db_path) as db:
        policy = db.execute("""SELECT p.retraction_state,p.source_version FROM source_policies p
            JOIN raw_messages m ON m.id=p.raw_id AND m.user_id=p.user_id WHERE m.content=?""", (STATEMENT,)).fetchone()
        receipt = db.execute("""SELECT e.source_version FROM atomic_extractions e
            JOIN raw_messages m ON m.id=e.source_raw_id WHERE m.content=?""", (STATEMENT,)).fetchone()
        assert policy == ("retracted", 2) and receipt[0] == 2
        assert db.execute("SELECT COUNT(*) FROM atomic_memories").fetchone()[0] >= 1
        if profile == "cycle2_dense":
            assert db.execute("SELECT COUNT(*) FROM atomic_vectors").fetchone()[0] > 0
    assert len(atoms.calls) == 1 and len(atoms.calls[0]) == 1


def test_later_retract_still_suppresses_without_reextracting(tmp_path):
    settings, atoms, vectors, application = build(tmp_path, "cycle2_raw", name="later.db")
    with TestClient(application) as client:
        assert client.post("/add", json=payload([STATEMENT], "original")).status_code == 200
        first = search(client, "tea")
        assert first.status_code == 200 and any(STATEMENT in item["content"] for item in first.json()["data"])
        calls = len(atoms.calls)
        retracted = client.post("/add", json=payload([RETRACT], "retract"))
        assert retracted.status_code == 200, retracted.json()
        later = search(client, "tea")
        assert later.status_code == 200 and later.json()["data"] == []
        assert len(atoms.calls) == calls == 1


def test_control_before_later_message_does_not_retract_it(tmp_path):
    settings, atoms, vectors, application = build(tmp_path, "cycle2_raw", name="order.db")
    with TestClient(application) as client:
        added = client.post("/add", json=payload([RETRACT, STATEMENT]))
        assert added.status_code == 200, added.json()
        found = search(client, "tea")
        assert found.status_code == 200
        assert found.json()["data"] and {item["content"] for item in found.json()["data"]} == {STATEMENT}
    assert len(atoms.calls) == 1 and len(atoms.calls[0]) == 1
    with sqlite3.connect(settings.db_path) as db:
        state = db.execute("""SELECT m.content,m.deletion_state,p.retraction_state,p.source_version
            FROM raw_messages m JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id
            ORDER BY m.message_index""").fetchall()
        assert state[1][1:] == ("active", "not_retracted", 1)


@pytest.mark.parametrize("failure", ["extract", "sql"])
def test_same_batch_injection_failure_publishes_nothing(tmp_path, failure, monkeypatch):
    settings, atoms, vectors, application = build(tmp_path, "cycle2_raw", name="failure.db")
    if failure == "extract":
        def fail_extract(messages, *, before_dispatch=None):
            if before_dispatch is not None:
                before_dispatch()
            raise am.AtomicMemoryFailure("offline_injected_failure")
        monkeypatch.setattr(atoms, "extract", fail_extract)
    else:
        with sqlite3.connect(settings.db_path) as db:
            db.execute("""CREATE TRIGGER fail_atom BEFORE INSERT ON atomic_memories
                BEGIN SELECT RAISE(ABORT,'offline_injected_failure'); END""")
    with TestClient(application) as client:
        added = client.post("/add", json=payload([STATEMENT, RETRACT]))
        assert added.status_code == 503
        if failure == "extract":
            assert added.json()["detail"] == "offline_injected_failure"
    assert_unpublished(settings.db_path)


@pytest.mark.parametrize("profile,enabled", [("cycle2_raw", False), ("cycle2_raw", True), ("cycle2_dense", True)])
def test_empty_health_is_ready_before_and_after_first_add(tmp_path, profile, enabled):
    settings, atoms, vectors, application = build(tmp_path, profile, enabled=enabled, name="health.db")
    with TestClient(application, raise_server_exceptions=False) as client:
        empty = client.get("/health")
        assert empty.status_code == 200, empty.text
        assert empty.json()["status"] == "ok"
        assert empty.json()["control_ready"] is True and empty.json()["data_route_ready"] is True
        assert client.post("/add", json=payload([STATEMENT])).status_code == 200
        populated = client.get("/health")
        assert populated.status_code == 200 and populated.json()["status"] == "ok"
        assert populated.json()["control_ready"] is True and populated.json()["data_route_ready"] is True


@pytest.mark.parametrize("gap", ["atom_receipt", "atom_fts", "span_dense"])
def test_missing_required_route_stays_unready_without_extra_dispatch(tmp_path, gap):
    enabled = gap != "span_dense"
    profile = "cycle2_raw" if enabled else "cycle2_dense"
    settings, atoms, vectors, application = build(tmp_path, profile, enabled=enabled, name="gap.db")
    with TestClient(application, raise_server_exceptions=False) as client:
        assert client.post("/add", json=payload([STATEMENT])).status_code == 200
        with sqlite3.connect(settings.db_path) as db:
            if gap == "atom_receipt":
                db.execute("DELETE FROM atomic_extractions")
            elif gap == "atom_fts":
                db.execute(f"DELETE FROM {am.partition_name('u')}")
            else:
                db.execute("DELETE FROM span_vectors")
        dispatch = atoms.calls if enabled else vectors.calls
        before = len(dispatch)
        health = client.get("/health")
        found = search(client, "tea")
        added = client.post("/add", json=payload(["another source."], "next"))
        expected = "incomplete_atom_route" if enabled else "incomplete_dense_route"
        assert health.status_code == 200
        assert health.json()["status"] == "not_ready" and health.json()["data_route_ready"] is False
        assert health.json()["reason"] == expected
        assert found.status_code == added.status_code == 503
        assert found.json()["detail"] == added.json()["detail"] == expected
        assert len(atoms.calls if enabled else vectors.calls) == before
