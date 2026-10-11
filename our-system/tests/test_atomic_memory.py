"""Local-only T08 source, publication, deletion, dispatch and isolation tests."""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest

from app import atomic_memory as am
from app.memory_core import initialize_core
from app.models import RawMessage
from app.store import SqliteStore


def raw(raw_id="r1", content="Alice avoids coffee when fasting.", *, user="u", stamp=None):
    return RawMessage(id=raw_id, user_id=user, session_id="s", request_id=raw_id,
                      message_index=0, role="user", content=content,
                      source_timestamp=stamp, recorded_at=1234)


def ref(message, quote=None):
    quote = message.content if quote is None else quote
    start = message.content.index(quote)
    return {"source_raw_id": message.id, "source_hash": am.source_hash(message.content),
            "start": start, "end": start + len(quote), "quote": quote}


def draft(message, *, kind="claim", text="navigation keyword", **fields):
    return {"kind": kind, "source_raw_id": message.id, "text": text,
            "source_refs": [ref(message)], **fields}


class StubProvider:
    implementation_kind = "test_stub"

    def __init__(self, items=None, *, config=None, payload=None):
        self.config = config or am.AtomicMemoryConfig("test", "https://mini.test/chat")
        self.items = items
        self.payload = payload
        self.calls = []

    def extract(self, messages, *, before_dispatch=None):
        if before_dispatch is not None:
            before_dispatch()
        self.calls.append([m.id for m in messages])
        items = [draft(m) for m in messages] if self.items is None else self.items
        return am.ExtractionBatch(self.payload if self.payload is not None else {"atoms": deepcopy(items)},
                                  {"requests": 1, "prompt_tokens": 10})


@contextmanager
def transaction(db):
    db.execute("BEGIN IMMEDIATE")
    with db:
        yield db


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "atoms.sqlite"
    SqliteStore(path).initialize()
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    with connection:
        am.initialize(connection)
    yield connection
    connection.close()


def save_raw(db, *messages):
    for message in messages:
        data = message.model_dump()
        columns = ",".join(data)
        placeholders = ",".join("?" for _ in data)
        db.execute(f"INSERT INTO raw_messages({columns}) VALUES ({placeholders})", tuple(data.values()))
    initialize_core(db)


def publish(db, messages, provider=None):
    provider = provider or StubProvider()
    prepared = am.prepare_atoms(messages, provider)
    assert prepared.status == "ready", prepared.error_code
    with transaction(db):
        save_raw(db, *messages)
        am.insert_atoms(db, prepared, user_id=messages[0].user_id)
    return prepared


def search(db, prepared, query="navigation", **kwargs):
    return am.search_atoms(db, "u", query, extractor_version=prepared.extractor_version, **kwargs)


def http_response(items, **updates):
    response = {"model": "gpt-4o-mini-2024-07-18", "choices": [{"finish_reason": "stop",
        "message": {"content": json.dumps({"atoms": items}, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 15, "total_tokens": 27}}
    response.update(updates)
    return response


def http_provider(monkeypatch, handler, **config):
    monkeypatch.setenv("TEST_ATOM_KEY", "local-placeholder-key")
    cfg = am.AtomicMemoryConfig("test-http", "https://mini.test/chat", api_key_env="TEST_ATOM_KEY", **config)
    return am.HttpAtomProvider(cfg, transport=httpx.MockTransport(handler))


def test_atomic_v1_identity_settings_and_implementation_are_bound():
    settings = SimpleNamespace(enable_extractor=True, profile_id="cycle2_raw", atom_endpoint="https://mini.test/chat",
        atom_model="gpt-4o-mini", atom_key_env="TEST_KEY", atom_batch_tokens=8192, atom_display=True)
    config = am.AtomicMemoryConfig.from_settings(settings)
    assert config.identity["version"] == "atomic-v1"
    assert config.api_key_env == "TEST_KEY" and config.display and config.max_batch_tokens == 8192
    stub = StubProvider(config=config)
    http = am.HttpAtomProvider(config)
    assert am.extractor_version(stub) == http.version == config.version
    assert http.identity["prompt_sha256"] == am.source_hash(am.EXTRACTION_PROMPT)
    assert am.extractor_version(StubProvider(config=replace(config, display=False))) != am.extractor_version(stub)
    settings.profile_id = "research"
    with pytest.raises(ValueError):
        am.AtomicMemoryConfig.from_settings(settings)


@pytest.mark.parametrize("change", [{"model": "other-mini"}, {"endpoint": "http://mini.test"},
    {"endpoint": "https://secret:password@mini.test"}, {"endpoint": "https://mini.test?key=secret"},
    {"timeout_seconds": float("nan")}, {"timeout_seconds": 0}, {"max_batch_tokens": True},
    {"max_batch_tokens": 8193}, {"max_batches": 0}])
def test_invalid_config(change):
    args = {"provider": "test", "endpoint": "https://mini.test/chat", **change}
    with pytest.raises(ValueError):
        am.AtomicMemoryConfig(**args)


def test_all_types_condition_negation_plan_and_procedure_are_navigation_only(db):
    message = raw(content="Bob dislikes caffeine when fasting. Next month he plans a trip. "
                          "Procedure v2: verify first, deploy second; skip deploy on failure.", stamp=0)
    items = [draft(message, kind=kind, text="caffeine navigation", subject="Bob", condition="when fasting",
                   scope="Bob", negated=True, plan="planned", time={"raw": "Next month", "precision": "relative"})
             for kind in ("claim", "event", "preference", "constraint", "procedure")]
    items[-1]["procedure"] = {"version": "v2", "prerequisites": ["verify"], "dependencies": ["deploy depends on verify"],
        "steps": ["verify", "deploy"], "exceptions": ["skip deploy on failure"], "completion_evidence": []}
    prepared = publish(db, [message], StubProvider(items))
    assert {a.draft.kind for a in prepared.atoms} == {"claim", "event", "preference", "constraint", "procedure"}
    assert all(a.draft.subject == "Bob" and a.draft.negated and a.draft.plan == "planned" for a in prepared.atoms)
    assert all(a.draft.time.start is None and a.draft.time.timezone is None for a in prepared.atoms)
    procedure = next(a.draft.procedure for a in prepared.atoms if a.draft.kind == "procedure")
    assert procedure.steps == ["verify", "deploy"] and procedure.completion_evidence == []
    result = search(db, prepared, "caffeine")
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.content == message.content and candidate.id == message.id
    assert candidate.representation == "raw" and candidate.created_at == 0
    assert candidate.source_sha256 == am.source_hash(message.content)
    assert result.trace["set_sufficient"] == "unknown" and len(result.trace["matched_atoms"]) == 5


def test_exact_unicode_offsets_hash_and_unknown_support(db):
    message = raw(content="😀\r\n张三说：不要咖啡。不要咖啡。")
    item = draft(message, text="coffee", subject="张三", negated=True)
    item["source_refs"] = [ref(message, "不要咖啡")]
    item["field_support"] = {"negated": {"state": "supported", "source_refs": [0]},
                              "subject": {"state": "unknown", "source_refs": []}}
    prepared = publish(db, [message], StubProvider([item]))
    atom = prepared.atoms[0]
    assert atom.source_valid and atom.field_supported == "unknown"
    assert atom.draft.field_support["value"].state == "unknown"
    assert atom.draft.value is None and atom.draft.time.start is None
    result = search(db, prepared, "coffee")
    citation = result.trace["matched_atoms"][0]["source_refs"][0]
    assert message.content[citation["start"]:citation["end"]] == "不要咖啡"
    assert citation["source_hash"] == am.source_hash(message.content)
    assert result.trace["matched_atoms"][0]["field_supported"] == "unknown"


@pytest.mark.parametrize("changes", [{"start": 999}, {"end": 999}, {"start": True}, {"end": 1},
    {"quote": "invented"}, {"source_raw_id": "missing"}, {"source_hash": "0" * 64},
    {"source_hash": "invalid"}])
def test_invalid_citations_fail_whole_batch(changes):
    message = raw(content="😀 coffee")
    item = draft(message)
    item["source_refs"][0].update(changes)
    prepared = am.prepare_atoms([message], StubProvider([draft(message), item]))
    assert prepared.status == "failed" and not prepared.atoms


def test_parent_hash_mismatch_rejected_before_dispatch():
    message = raw()
    provider = StubProvider()
    prepared = am.prepare_atoms([message], provider, source_hashes={message.id: "0" * 64})
    assert prepared.error_code == "atom_source_hash_mismatch" and provider.calls == []
    assert am.prepare_atoms([message], provider, source_hashes={message.id: am.source_hash(message.content)}).status == "ready"


@pytest.mark.parametrize("support", [{"made_up": {"state": "unknown"}},
    {"subject": {"state": "supported", "source_refs": [0]}},
    {"text": {"state": "supported", "source_refs": []}},
    {"text": {"state": "supported", "source_refs": [1]}},
    {"text": {"state": "supported", "source_refs": [True]}}])
def test_field_support_cannot_create_missing_values_or_refs(support):
    message = raw()
    prepared = am.prepare_atoms([message], StubProvider([draft(message, field_support=support)]))
    assert prepared.status == "failed" and not prepared.atoms


def test_primary_ref_unknown_schema_and_procedure_errors():
    message, other = raw(), raw("r2")
    wrong = draft(message, source_refs=[ref(other)])
    assert am.prepare_atoms([message, other], StubProvider([wrong])).error_code == "atom_primary_source_uncited"
    wrong = draft(message, unexpected="ignored?")
    assert am.prepare_atoms([message], StubProvider([wrong])).error_code == "atom_invalid_schema"
    wrong = draft(message, procedure={"steps": ["deploy"]})
    assert am.prepare_atoms([message], StubProvider([wrong])).error_code == "atom_invalid_procedure"


@pytest.mark.parametrize("messages", [[raw(), raw("r2", user="other")], [raw(), raw()],
    [raw().model_copy(update={"deletion_state": "deleted"})], [raw(content=" ")]])
def test_same_user_active_unique_sources_preflight(messages):
    provider = StubProvider()
    prepared = am.prepare_atoms(messages, provider)
    assert prepared.error_code == "atom_invalid_sources" and provider.calls == []


def test_batch_bounds_are_on_prompt_plus_full_sources_before_any_dispatch():
    first = raw(content="😀 short")
    second = raw("r2", "中文" * 2000)
    config = am.AtomicMemoryConfig("test", "https://mini.test/chat", max_batch_tokens=600,
                                   max_batch_characters=96000, max_batch_messages=1)
    provider = StubProvider(config=config)
    assert am.prepare_atoms([first, second], provider).error_code == "atom_source_exceeds_batch"
    assert not provider.calls
    exact = am.batch_token_count([first])
    assert len(am.build_batches([first], replace(config, max_batch_tokens=exact))) == 1
    assert am.prepare_atoms([first], StubProvider(config=replace(config, max_batch_tokens=exact - 1))).status == "failed"
    many = [raw(f"r{i}", "text") for i in range(3)]
    provider = StubProvider(config=replace(config, max_batches=2))
    assert am.prepare_atoms(many, provider).error_code == "atom_input_limit" and not provider.calls


def test_late_batch_failure_has_no_partial_atoms_and_keeps_attempt_cost():
    class BrokenProvider(StubProvider):
        def extract(self, messages, *, before_dispatch=None):
            if self.calls:
                raise am.AtomicMemoryFailure("atom_timeout", {"requests": 1, "usage_complete": False})
            return super().extract(messages, before_dispatch=before_dispatch)
    provider = BrokenProvider(config=am.AtomicMemoryConfig("test", "https://mini.test/chat", max_batch_messages=1))
    prepared = am.prepare_atoms([raw(), raw("r2")], provider)
    assert prepared.status == "failed" and prepared.atoms == ()
    assert len(prepared.usage["batches"]) == 2 and prepared.error_code == "atom_timeout"


def test_http_json_wire_and_guard_before_each_batch(monkeypatch):
    events = []
    def handler(request):
        events.append("http")
        body = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer local-placeholder-key"
        assert body["model"] == "gpt-4o-mini" and body["response_format"] == {"type": "json_object"}
        assert body["max_tokens"] == 8192
        source = json.loads(body["messages"][1]["content"])["messages"][0]
        assert source["source_hash"] == am.source_hash(source["content"])
        return httpx.Response(200, json=http_response([]))
    provider = http_provider(monkeypatch, handler, max_batch_messages=1)
    prepared = am.prepare_atoms([raw(), raw("r2")], provider, before_dispatch=lambda: events.append("guard"))
    assert prepared.status == "ready" and events == ["guard", "http", "guard", "http"]
    assert len(prepared.sources) == 2 and not prepared.atoms
    assert prepared.usage["batches"][0]["returned_model"] == "gpt-4o-mini-2024-07-18"


@pytest.mark.parametrize("guard_error", [RuntimeError("lease_lost"), ValueError("source_revoked")])
def test_guard_exception_propagates_without_sending(monkeypatch, guard_error):
    events = []
    provider = http_provider(monkeypatch, lambda request: events.append("http"))
    def guard():
        raise guard_error
    with pytest.raises(type(guard_error), match=str(guard_error)):
        am.prepare_atoms([raw()], provider, before_dispatch=guard)
    assert events == []


def test_missing_credentials_and_missing_guard_never_dispatch(monkeypatch):
    events = []
    provider = http_provider(monkeypatch, lambda request: events.append("http"))
    assert am.prepare_atoms([raw()], provider).error_code == "atom_missing_dispatch_guard"
    monkeypatch.delenv("TEST_ATOM_KEY")
    assert am.prepare_atoms([raw()], provider, before_dispatch=lambda: None).error_code == "atom_missing_api_key"
    assert not events


@pytest.mark.parametrize("failure", [429, 500, 503, 302, "timeout", "transport"])
def test_http_failures_are_one_attempt_and_safe(monkeypatch, failure):
    calls = []
    def handler(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private body", request=request)
        if failure == "transport":
            raise httpx.ConnectError("private key", request=request)
        return httpx.Response(failure, text="sensitive error body", headers={"Location": "https://elsewhere.test"})
    provider = http_provider(monkeypatch, handler)
    prepared = am.prepare_atoms([raw()], provider, before_dispatch=lambda: None)
    assert prepared.status == "failed" and len(calls) == 1
    assert "private" not in str(prepared) and "sensitive error" not in str(prepared)
    assert prepared.usage["batches"][0]["requests"] == 1


@pytest.mark.parametrize("case", ["length", "refusal", "tools", "model", "missing_model", "malformed", "duplicate", "nan", "count"])
def test_invalid_incomplete_or_wrong_model_response_never_succeeds(monkeypatch, case):
    data = http_response([])
    if case == "length": data["choices"][0]["finish_reason"] = "length"
    if case == "refusal": data["choices"][0]["message"]["refusal"] = "refused"
    if case == "tools": data["choices"][0]["message"]["tool_calls"] = [{}]
    if case == "model": data["model"] = "gpt-4o"
    if case == "missing_model": del data["model"]
    if case == "malformed": data["choices"][0]["message"]["content"] = '{"atoms":'
    if case == "duplicate": data["choices"][0]["message"]["content"] = '{"atoms":[],"atoms":[]}'
    if case == "nan": data["choices"][0]["message"]["content"] = '{"atoms":[],"value":NaN}'
    if case == "count": data["choices"] = []
    provider = http_provider(monkeypatch, lambda _: httpx.Response(200, json=data))
    assert am.prepare_atoms([raw()], provider, before_dispatch=lambda: None).status == "failed"


def test_bounded_http_response_and_atom_count(monkeypatch):
    provider = http_provider(monkeypatch, lambda _: httpx.Response(200, content=b"x" * 1024), max_response_bytes=512)
    assert am.prepare_atoms([raw()], provider, before_dispatch=lambda: None).error_code == "atom_response_limit"
    message = raw()
    stub = StubProvider([draft(message)] * 2,
        config=am.AtomicMemoryConfig("test", "https://mini.test/chat", max_atoms_per_batch=1))
    assert am.prepare_atoms([message], stub).error_code == "atom_output_limit"


def test_optional_provider_and_empty_active_input():
    assert am.prepare_atoms([raw()]).status == "unavailable"
    provider = StubProvider()
    assert am.prepare_atoms([], provider).status == "ready" and provider.calls == []
    assert am.prepare_atoms([]).status == "ready"


def test_receipts_count_completed_raw_even_with_zero_atoms(db):
    messages = [raw(), raw("r2", "No durable information.")]
    prepared = publish(db, messages, StubProvider([draft(messages[0])]))
    assert am.counts(db, "u", prepared.extractor_version) == {"active": 2, "raw": 2, "atom": 2, "fts": 2, "atoms": 1}
    with transaction(db):
        assert am.insert_atoms(db, prepared, user_id="u") == 0
    assert db.execute("SELECT COUNT(*) FROM atomic_extractions").fetchone()[0] == 2
    assert db.execute(f"SELECT COUNT(*) FROM {am.partition_name('u')}").fetchone()[0] == 1


def test_empty_extraction_receipt_is_bound_to_identity_and_source(db):
    message = raw()
    prepared = publish(db, [message], StubProvider([]))
    assert am.counts(db, "u", prepared.extractor_version)["atom"] == 1
    assert search(db, prepared).candidates == []
    with transaction(db):
        db.execute("UPDATE atomic_extractions SET extractor_json='{}'")
    assert am.counts(db, "u", prepared.extractor_version)["atom"] == 0


def test_missing_corrupt_or_duplicate_fts_prevents_ready(db):
    prepared = publish(db, [raw()])
    table = am.partition_name("u")
    with transaction(db):
        db.execute(f"UPDATE {table} SET lexical_text='corrupt navigation'")
    assert am.counts(db, "u", prepared.extractor_version)["fts"] == 0
    assert search(db, prepared).candidates == []
    assert search(db, prepared).candidates == []
    with transaction(db):
        db.execute(f"DELETE FROM {table}")
        db.execute(f"INSERT INTO {table} VALUES (?,?)", (prepared.atoms[0].id, am._navigation_text(prepared.atoms[0].draft)))
        db.execute(f"INSERT INTO {table} SELECT * FROM {table}")
    assert am.counts(db, "u", prepared.extractor_version)["fts"] == 0


def test_same_source_extractor_replay_cannot_append_different_atoms(db):
    message = raw()
    prepared = publish(db, [message])
    other = am.prepare_atoms([message], StubProvider([draft(message, text="different navigation")]))
    with transaction(db):
        with pytest.raises(am.AtomicMemoryFailure, match="atom_replay_conflict"):
            am.insert_atoms(db, other, user_id="u")
    assert [r[0] for r in db.execute("SELECT id FROM atomic_memories")] == [prepared.atoms[0].id]
    different = am.prepare_atoms([message], StubProvider(config=replace(StubProvider().config, revision="2")))
    with transaction(db):
        assert am.insert_atoms(db, different, user_id="u") == 1
    assert len(search(db, different).candidates) == 1
    assert search(db, different).trace["extractor_version"] != prepared.extractor_version


def test_mutations_require_parent_transaction_and_failed_preparation_not_empty_success(db):
    prepared = am.prepare_atoms([raw()], StubProvider())
    for action in (lambda: am.insert_atoms(db, prepared, user_id="u"),
                   lambda: am.delete_atoms(db, "u", ["r1"]), lambda: am.invalidate_atoms(db, "u", ["r1"]),
                   lambda: am.prune(db, "u")):
        with pytest.raises(am.AtomicMemoryFailure, match="atom_transaction_required"):
            action()
    db.execute("BEGIN")
    with pytest.raises(am.AtomicMemoryFailure, match="atom_missing_configuration"):
        am.insert_atoms(db, am.prepare_atoms([raw()]), user_id="u")
    db.rollback()


def test_parent_rollback_removes_raw_atom_fts_and_receipts(db):
    message = raw()
    prepared = am.prepare_atoms([message], StubProvider())
    db.execute("BEGIN IMMEDIATE")
    save_raw(db, message)
    am.insert_atoms(db, prepared, user_id="u")
    assert len(search(db, prepared).candidates) == 1
    db.rollback()
    assert db.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM atomic_memories").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM atomic_extractions").fetchone()[0] == 0
    assert search(db, prepared).candidates == []


def test_atom_internal_savepoint_does_not_commit_parent_or_leave_partial_rows(db):
    messages = [raw(), raw("r2")]
    prepared = am.prepare_atoms(messages, StubProvider())
    db.execute("BEGIN")
    save_raw(db, *messages)
    second = prepared.atoms[1].id
    db.execute(f"""CREATE TRIGGER fail_atom BEFORE INSERT ON atomic_memories
        WHEN NEW.id='{second}' BEGIN SELECT RAISE(ABORT,'injected'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        am.insert_atoms(db, prepared, user_id="u")
    assert db.in_transaction and db.execute("SELECT COUNT(*) FROM atomic_memories").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM atomic_extractions").fetchone()[0] == 0
    db.rollback()


@pytest.mark.parametrize("tamper", ["content", "hash", "version", "user", "visibility", "policy", "deleted", "retracted"])
def test_source_changes_reject_late_commit_and_search_and_prune(db, tamper):
    prepared = publish(db, [raw()])
    with transaction(db):
        sql = {"content": "UPDATE raw_messages SET content='changed'", "hash": "UPDATE source_policies SET content_sha256='bad'",
            "version": "UPDATE source_policies SET source_version=2", "user": "UPDATE source_policies SET user_id='other'",
            "visibility": "UPDATE source_policies SET visibility='suppressed'", "policy": "UPDATE source_policies SET disclosure_policy='blocked'",
            "deleted": "UPDATE raw_messages SET deletion_state='deleted',content=''",
            "retracted": "UPDATE source_policies SET retraction_state='retracted',source_version=2"}[tamper]
        db.execute(sql)
    assert search(db, prepared).candidates == []
    expected_before = 1 if tamper in {"visibility", "policy"} else 0
    assert am.counts(db, "u", prepared.extractor_version)["atom"] == expected_before
    with transaction(db):
        with pytest.raises(am.AtomicMemoryFailure, match="atom_source_changed"):
            am.insert_atoms(db, prepared, user_id="u")
        retained = tamper in {"version", "visibility", "policy", "retracted"}
        assert am.prune(db, "u") == (0 if retained else 1)
    assert db.execute("SELECT COUNT(*) FROM atomic_memories").fetchone()[0] == int(retained)
    assert db.execute("SELECT COUNT(*) FROM atomic_extractions").fetchone()[0] == int(retained)
    assert db.execute(f"SELECT COUNT(*) FROM {am.partition_name('u')}").fetchone()[0] == int(retained)
    assert am.counts(db, "u", prepared.extractor_version)["atom"] == int(retained)


def test_linked_dependency_erasure_deletes_whole_atom_and_blocks_late_replay(db):
    first, second = raw(), raw("r2", "Bob confirms the condition.")
    item = draft(first, source_refs=[ref(first), ref(second)])
    prepared = publish(db, [first, second], StubProvider([item]))
    assert {h.id for h in search(db, prepared).candidates} == {"r1", "r2"}
    with transaction(db):
        assert am.delete_atoms(db, "u", ["r2"]) == 1
        assert am.delete_atoms(db, "u", ["r2"]) == 0
    assert search(db, prepared).candidates == []
    assert db.execute("SELECT COUNT(*) FROM atomic_source_refs").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM atomic_extractions").fetchone()[0] == 1
    assert db.execute("SELECT source_raw_id FROM atomic_extractions").fetchone()[0] == "r1"
    assert "Bob" not in str([tuple(r) for r in db.execute("SELECT * FROM atomic_source_blocks")])
    with transaction(db):
        with pytest.raises(am.AtomicMemoryFailure, match="atom_source_changed"):
            am.insert_atoms(db, prepared, user_id="u")


def test_zero_atom_invalidation_and_erasure_also_block_late_receipt(db):
    prepared = publish(db, [raw()], StubProvider([]))
    with transaction(db):
        assert am.invalidate_atoms(db, "u", ["r1"]) == 0
    assert am.counts(db, "u", prepared.extractor_version)["atom"] == 0
    with transaction(db):
        with pytest.raises(am.AtomicMemoryFailure, match="atom_source_changed"):
            am.insert_atoms(db, prepared, user_id="u")
    assert db.execute("SELECT COUNT(*) FROM atomic_extractions").fetchone()[0] == 0


def test_invalidation_keeps_marked_payload_then_prune_scrubs_it(db):
    prepared = publish(db, [raw()])
    with transaction(db):
        assert am.invalidate_atoms(db, "u", ["r1"], extractor_version=prepared.extractor_version) == 1
    assert db.execute("SELECT state FROM atomic_memories").fetchone()[0] == "invalidated"
    assert search(db, prepared).candidates == []
    with transaction(db):
        assert am.prune(db, "u") == 1
    assert db.execute("SELECT COUNT(*) FROM atomic_memories").fetchone()[0] == 0


def test_prune_cleans_cascaded_raw_delete_zero_receipt_and_orphan_fts(db):
    prepared = publish(db, [raw(), raw("r2")], StubProvider([draft(raw())]))
    with transaction(db):
        db.execute("DELETE FROM raw_messages WHERE user_id='u'")
        am.prune(db, "u")
    assert db.execute("SELECT COUNT(*) FROM atomic_source_refs").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM atomic_extractions").fetchone()[0] == 0
    assert db.execute(f"SELECT COUNT(*) FROM {am.partition_name('u')}").fetchone()[0] == 0
    assert search(db, prepared).candidates == []


def test_user_isolation_fts_statistics_and_deletion_are_partitioned(db):
    prepared = publish(db, [raw(), raw("r2", "tea")])
    before = [(h.id, h.score) for h in search(db, prepared).candidates]
    other = [raw(f"other{i}", "navigation keyword coffee", user="other") for i in range(30)]
    other_prepared = publish(db, other)
    after = [(h.id, h.score) for h in search(db, prepared).candidates]
    assert before == after
    result = am.search_atoms(db, "other", "navigation", extractor_version=other_prepared.extractor_version)
    assert all(hit.user_id == "other" for hit in result.candidates)
    with transaction(db):
        assert am.delete_atoms(db, "other", ["r1"]) == 0
    assert before == [(h.id, h.score) for h in search(db, prepared).candidates]
    with transaction(db):
        with pytest.raises(am.AtomicMemoryFailure, match="atom_invalid_sources"):
            am.insert_atoms(db, prepared, user_id="other")


def test_restricted_requires_explicit_permission_and_trace_does_not_leak_dropped_raw(db):
    messages = [raw(), raw("r2")]
    prepared = publish(db, messages, StubProvider([draft(messages[0], source_refs=[ref(m) for m in messages])]))
    result = search(db, prepared, limit=1)
    assert len(result.candidates) == 1 and result.trace["matched_atoms"] == []
    with transaction(db):
        db.execute("UPDATE source_policies SET disclosure_policy='restricted' WHERE raw_id='r2'")
    assert search(db, prepared).candidates == []
    assert am.counts(db, "u", prepared.extractor_version)["active"] == 2
    assert am.counts(db, "u", prepared.extractor_version)["atom"] == 2
    allowed = frozenset({"r2"})
    assert len(search(db, prepared, allowed_restricted_ids=allowed).candidates) == 2
    assert am.counts(db, "u", prepared.extractor_version, allowed_restricted_ids=allowed)["atom"] == 2


def test_payload_hash_extractor_and_dependency_tampering_not_returned(db):
    prepared = publish(db, [raw()])
    atom = prepared.atoms[0]
    tampered = json.loads(db.execute("SELECT payload_json FROM atomic_memories").fetchone()[0])
    tampered["draft"]["text"] = "invented navigation"
    with transaction(db):
        db.execute("UPDATE atomic_memories SET payload_json=?", (json.dumps(tampered),))
    assert search(db, prepared).candidates == []
    assert am.counts(db, "u", prepared.extractor_version)["atom"] == 0
    with transaction(db):
        db.execute("UPDATE atomic_memories SET payload_json=?", (am._json(atom.model_dump(mode="json")),))
        db.execute("DELETE FROM atomic_source_refs")
    assert search(db, prepared).candidates == []


def test_unknown_query_no_recent_fallback_and_limits_apply_to_raws(db):
    prepared = publish(db, [raw(), raw("r2")])
    assert search(db, prepared, "absentnohit").candidates == []
    assert search(db, prepared, "!").candidates == []
    assert len(search(db, prepared, limit=1, candidate_limit=1).candidates) == 1
    assert not db.in_transaction
    db.execute("BEGIN")
    search(db, prepared)
    assert db.in_transaction
    db.rollback()


def test_explicit_version_gate_and_receipt_cannot_be_forged_with_missing_atom(db):
    prepared = publish(db, [raw()])
    with transaction(db):
        with pytest.raises(am.AtomicMemoryFailure, match="atom_extractor_mismatch"):
            am.insert_atoms(db, prepared, user_id="u", expected_extractor_version="other")
        db.execute("DELETE FROM atomic_memories")
    assert am.counts(db, "u", prepared.extractor_version)["atom"] == 0
    assert am.counts(db, "u", "other")["atom"] == 0
    assert search(db, prepared).candidates == []


def test_parent_profile_config_http_and_injected_provider_have_one_version(tmp_path):
    from app.config import Settings
    from app.profiles import Profile

    settings = Settings(db_path=tmp_path / "profile.sqlite", profile_id="cycle2_raw",
                        enable_extractor=True, atom_endpoint="https://mini.test/chat")
    config = am.AtomicMemoryConfig.from_settings(settings)
    stub = StubProvider(config=config)
    am.validate_provider(stub, config)
    http = am.HttpAtomProvider(config)
    profile = Profile.build(settings, None)
    prepared = am.prepare_atoms([raw()], stub)
    assert profile.identity["atom_extractor_version"] == config.version == http.version == prepared.extractor_version
    assert prepared.metadata == profile.identity["atom"]
    assert "implementation_kind" not in prepared.metadata
    assert prepared.usage["implementation_kind"] == "test_stub"
    with pytest.raises(ValueError, match="identity"):
        am.validate_provider(StubProvider(config=replace(config, revision="other")), config)
    with pytest.raises(ValueError, match="identity"):
        am.validate_provider(StubProvider(), config)


def test_display_defaults_off_and_labels_derived_together_with_all_raw(db):
    first, second = raw(), raw("r2", "Bob confirms this procedure.")
    item = draft(first, kind="procedure", text="verify then deploy", subject="Bob",
        procedure={"version": "v2", "steps": ["verify", "deploy"], "exceptions": ["failure"]},
        source_refs=[ref(first), ref(second)])
    prepared = publish(db, [first, second], StubProvider([item]))
    hits = search(db, prepared, "verify").candidates
    disabled = am.atom_support_groups(db, "u", hits, extractor_version=prepared.extractor_version)
    assert disabled.groups == () and not disabled.trace["enabled"]
    enabled = am.atom_support_groups(db, "u", hits, extractor_version=prepared.extractor_version, enabled=True)
    assert len(enabled.groups) == 1
    group = enabled.support_groups[0]
    assert group["seed_id"] == first.id and set(group["member_ids"]) == {first.id, second.id}
    assert group["semantic_entailment"] == "unknown"
    rendered = am.render_atoms_with_raw(hits, enabled.groups, raw_renderer=lambda hit: "[role=" + hit.role + "]\n" + hit.content)
    assert "Derived navigation atom" in rendered[first.id] and "not a Raw quotation" in rendered[first.id]
    assert '"steps":["verify","deploy"]' in rendered[first.id]
    assert rendered[first.id].endswith("[role=user]\n" + first.content)
    assert rendered[second.id] == "[role=user]\n" + second.content
    assert {h.id: h.content for h in hits} == {first.id: first.content, second.id: second.content}


def test_display_omits_missing_changed_wrong_user_or_budgeted_dependency(db):
    first, second = raw(), raw("r2", "context")
    prepared = publish(db, [first, second], StubProvider([draft(first, source_refs=[ref(first), ref(second)])]))
    hits = search(db, prepared).candidates
    enabled = am.atom_support_groups(db, "u", hits, extractor_version=prepared.extractor_version, enabled=True)
    partial = [hit for hit in hits if hit.id == "r1"]
    assert am.atom_support_groups(db, "u", partial, extractor_version=prepared.extractor_version, enabled=True).groups == ()
    assert am.render_atoms_with_raw(partial, enabled.groups) == {"r1": first.content}
    changed = [h.model_copy(update={"content": "changed"}) if h.id == "r2" else h for h in hits]
    assert am.render_atoms_with_raw(changed, enabled.groups)["r1"] == first.content
    mixed_user = [h.model_copy(update={"user_id": "other"}) if h.id == "r2" else h for h in hits]
    assert am.render_atoms_with_raw(mixed_user, enabled.groups)["r1"] == first.content
    budgeted = am.atom_support_groups(db, "u", hits, extractor_version=prepared.extractor_version,
                                     enabled=True, max_derived_characters=1)
    assert not budgeted.groups
    with transaction(db):
        db.execute("UPDATE source_policies SET visibility='suppressed' WHERE raw_id='r2'")
    assert am.atom_support_groups(db, "u", hits, extractor_version=prepared.extractor_version, enabled=True).groups == ()


def test_display_separate_user_atom_and_id_scope(db):
    prepared = publish(db, [raw()])
    hits = search(db, prepared).candidates
    assert am.atom_support_groups(db, "other", hits, extractor_version=prepared.extractor_version, enabled=True).groups == ()
    assert am.atom_support_groups(db, "u", hits, extractor_version="other", enabled=True).groups == ()
    assert am.atom_support_groups(db, "u", hits, extractor_version=prepared.extractor_version, enabled=True, atom_ids=[]).groups == ()
    selected = am.atom_support_groups(db, "u", hits, extractor_version=prepared.extractor_version, enabled=True,
                                     atom_ids=[prepared.atoms[0].id])
    assert len(selected.groups) == 1


def test_prune_empty_receipt_after_raw_erasure(db):
    prepared = publish(db, [raw()], StubProvider([]))
    with transaction(db):
        db.execute("UPDATE raw_messages SET content='',deletion_state='deleted'")
        assert am.prune(db, "u") == 0
    assert db.execute("SELECT COUNT(*) FROM atomic_extractions").fetchone()[0] == 0
    assert am.counts(db, "u", prepared.extractor_version) == {"active": 0, "raw": 0, "atom": 0, "fts": 0, "atoms": 0}


def test_init_and_read_helpers_support_plain_sqlite_rows_and_never_commit(db):
    prepared = publish(db, [raw()])
    db.row_factory = None
    assert len(search(db, prepared).candidates) == 1
    assert am.counts(db, "u", prepared.extractor_version)["atom"] == 1
    db.execute("BEGIN")
    am.initialize(db)
    assert db.in_transaction
    db.rollback()


def test_raw_privacy_level_cannot_bypass_disclosure_policy(db):
    prepared = publish(db, [raw()])
    hits = search(db, prepared).candidates
    with transaction(db):
        db.execute("UPDATE raw_messages SET privacy_level='restricted'")
    assert search(db, prepared).candidates == []
    assert am.counts(db, "u", prepared.extractor_version)["active"] == 1
    assert am.counts(db, "u", prepared.extractor_version)["atom"] == 1
    assert am.atom_support_groups(db, "u", hits, extractor_version=prepared.extractor_version, enabled=True).groups == ()
    assert len(search(db, prepared, allowed_restricted_ids=frozenset({"r1"})).candidates) == 1


def test_explicit_version_preparation_and_tombstone_prevent_resurrection(db):
    message = raw()
    prepared = am.prepare_atoms([message], StubProvider(), source_versions={"r1": 2})
    assert prepared.atoms[0].source_version == 2
    with transaction(db):
        save_raw(db, message)
        db.execute("UPDATE source_policies SET source_version=2")
        assert am.insert_atoms(db, prepared, user_id="u") == 1
    assert len(search(db, prepared).candidates) == 1
    with transaction(db):
        db.execute("INSERT INTO memory_tombstones VALUES (?,?,?,?)", ("u", "r1", am.statement_digest(message.content), 1))
    assert search(db, prepared).candidates == []
    with transaction(db):
        with pytest.raises(am.AtomicMemoryFailure, match="atom_source_changed"):
            am.insert_atoms(db, prepared, user_id="u")
        am.prune(db, "u")
    assert db.execute("SELECT COUNT(*) FROM atomic_extractions").fetchone()[0] == 0


def test_parent_cycle2_leased_add_search_replay_and_delete_local_http(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.api import create_app
    from app.config import Settings

    settings = Settings(db_path=tmp_path / "integrated.sqlite", profile_id="cycle2_raw", enable_extractor=True,
                        atom_endpoint="https://mini.test/chat", enable_governance=True)
    config = am.AtomicMemoryConfig.from_settings(settings)
    calls = []
    monkeypatch.setenv(config.api_key_env, "local-placeholder-key")

    def handler(request):
        calls.append(request)
        messages = json.loads(json.loads(request.content)["messages"][1]["content"])["messages"]
        items = []
        for source in messages:
            message = raw(source["source_raw_id"], source["content"])
            items.append(draft(message, text="atomnavigationmarker"))
        return httpx.Response(200, json=http_response(items))

    provider = am.HttpAtomProvider(config, transport=httpx.MockTransport(handler))
    am.validate_provider(provider, config)
    with TestClient(create_app(settings, atom_provider=provider)) as client:
        payload = {"user_id": "u", "session_id": "s", "request_id": "add",
                   "messages": [{"role": "user", "content": "I enjoy coffee in the morning."}]}
        assert client.post("/add", json=payload).status_code == 200
        assert client.post("/add", json=payload).status_code == 200 and len(calls) == 1
        response = client.post("/search", json={"user_id": "u", "query": "atomnavigationmarker", "top_k": 20})
        assert response.status_code == 200, response.text
        assert [hit["content"] for hit in response.json()["data"]] == [payload["messages"][0]["content"]]
        assert client.post("/search", json={"user_id": "other", "query": "atomnavigationmarker", "top_k": 20}).json()["data"] == []
        erased = {"user_id": "u", "session_id": "s", "request_id": "delete",
                  "messages": [{"role": "user", "content": "请忘记“I enjoy coffee in the morning.”"}]}
        deletion = client.post("/add", json=erased)
        assert deletion.status_code == 200, deletion.text
        assert len(calls) == 1
        assert client.post("/search", json={"user_id": "u", "query": "atomnavigationmarker", "top_k": 20}).json()["data"] == []
        with sqlite3.connect(settings.db_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM atomic_memories").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM atomic_extractions").fetchone()[0] == 0


def vector_batch(prepared, vectors=None, *, revision="1"):
    from app.embeddings import EmbeddingBatch, EmbeddingConfig, prepare_embedding

    class LocalEmbedding:
        implementation_kind = "test_double"
        config = EmbeddingConfig(provider="test-atom", endpoint="https://embedding.test/embeddings",
                                 model="test-vector", dimensions=2, revision=revision)

        def embed(self, texts):
            values = vectors if vectors is not None else [[1.0, 0.0] if "east" in text else [0.0, 1.0] for text in texts]
            return EmbeddingBatch(values, {"requests": 1})

    return prepare_embedding(LocalEmbedding(), am.prepared_atom_texts(prepared))


def dense_search(db, prepared, embedding, vector=None, **kwargs):
    return am.search_atom_dense(db, "u", vector or [1.0, 0.0], embedding.version, embedding.dimensions,
                                prepared.extractor_version, **kwargs)


def test_prepared_atom_texts_stable_sort_preserves_typed_fields():
    message = raw()
    prepared = am.prepare_atoms([message], StubProvider([
        draft(message, kind="preference", text="east", condition="fasting", negated=True),
        draft(message, kind="procedure", text="north", procedure={"steps": ["verify", "deploy"]})]))
    texts = am.prepared_atom_texts(prepared)
    assert texts == am.prepared_atom_texts(replace(prepared, atoms=tuple(reversed(prepared.atoms))))
    assert len(texts) == 2 and any('"negated":true' in text for text in texts)
    assert any('"steps":["verify","deploy"]' in text for text in texts)
    assert all("source_raw_id" not in text and "source_hash" not in text for text in texts)


def test_atom_dense_returns_same_raw_binding_with_exact_score_and_version(db):
    first, second = raw("r1", "one complete Raw"), raw("r2", "another complete Raw")
    prepared = publish(db, [first, second], StubProvider([draft(first, text="east"), draft(second, text="north")]))
    embedding = vector_batch(prepared)
    with transaction(db):
        assert am.insert_atom_vectors(db, prepared, embedding) == 2
        assert am.insert_atom_vectors(db, replace(prepared, atoms=tuple(reversed(prepared.atoms))), embedding) == 0
    result = dense_search(db, prepared, embedding)
    assert [(h.id, h.content, h.score) for h in result.candidates] == [(first.id, first.content, 1.0)]
    assert result.candidates[0].source_sha256 == am.source_hash(first.content)
    assert result.trace["matched_atoms"][0]["source_valid"]
    assert result.trace["embedding_version"] == embedding.version
    assert result.trace["representation"] == "raw" and result.trace["set_sufficient"] == "unknown"
    readiness = am.counts(db, "u", prepared.extractor_version, embedding_version=embedding.version, dimensions=2)
    assert readiness["atom"] == readiness["fts"] == readiness["dense"] == 2
    assert am.counts(db, "u", prepared.extractor_version, embedding_version="other", dimensions=2)["dense"] == 0


def test_dense_multi_atom_max_score_not_multiplicity_and_shared_limit(db):
    first, second = raw(), raw("r2", "second source")
    prepared = publish(db, [first, second], StubProvider([
        draft(first, text="east first"), draft(first, text="east second"), draft(second, text="east third")]))
    embedding = vector_batch(prepared)
    with transaction(db):
        am.insert_atom_vectors(db, prepared, embedding, user_id="u")
    result = dense_search(db, prepared, embedding)
    assert len(result.candidates) == 2 and all(h.score == 1.0 for h in result.candidates)
    bounded = dense_search(db, prepared, embedding, limit=1, candidate_limit=1)
    assert len(bounded.candidates) == len(bounded.trace["matched_atoms"]) == 1
    assert am.search_atom_dense(db, "other", [1.0, 0.0], embedding.version, 2,
                                prepared.extractor_version).candidates == []
    assert dense_search(db, prepared, replace(embedding, version="other")).candidates == []


def test_dense_multisource_and_zero_atom_readiness(db):
    first, second, third = raw(), raw("r2", "context"), raw("r3", "no facts")
    prepared = publish(db, [first, second, third], StubProvider([draft(first, text="east", source_refs=[ref(first), ref(second)])]))
    embedding = vector_batch(prepared)
    with transaction(db):
        am.insert_atom_vectors(db, prepared, embedding)
    dense = dense_search(db, prepared, embedding)
    lexical = search(db, prepared, "east")
    assert {h.id for h in dense.candidates} == {h.id for h in lexical.candidates} == {first.id, second.id}
    assert dense.trace["matched_atoms"][0]["source_bindings"] == lexical.trace["matched_atoms"][0]["source_bindings"]
    assert am.counts(db, "u", prepared.extractor_version, embedding_version=embedding.version, dimensions=2)["dense"] == 3


def test_zero_atoms_do_not_require_any_vectors(db):
    prepared = publish(db, [raw()], StubProvider([]))
    embedding = vector_batch(prepared, [])
    assert am.prepared_atom_texts(prepared) == []
    with transaction(db):
        assert am.insert_atom_vectors(db, prepared, embedding) == 0
    assert am.counts(db, "u", prepared.extractor_version, embedding_version=embedding.version, dimensions=2)["dense"] == 1
    assert dense_search(db, prepared, embedding).candidates == []


@pytest.mark.parametrize("invalid", [[0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0], [True, 1.0], [1.0]])
def test_atom_vectors_reject_invalid_numeric_batches_atomically(db, invalid):
    prepared = publish(db, [raw()])
    embedding = vector_batch(prepared)
    malformed = replace(embedding, vectors=[invalid])
    with transaction(db):
        with pytest.raises(am.AtomicMemoryFailure):
            am.insert_atom_vectors(db, prepared, malformed)
    assert db.execute("SELECT COUNT(*) FROM atomic_vectors").fetchone()[0] == 0


def test_vectors_require_matching_ready_embedding_identity_and_existing_atoms(db):
    prepared = am.prepare_atoms([raw()], StubProvider())
    embedding = vector_batch(prepared)
    with transaction(db):
        save_raw(db, raw())
        with pytest.raises(am.AtomicMemoryFailure, match="missing_receipt"):
            am.insert_atom_vectors(db, prepared, embedding)
        am.insert_atoms(db, prepared, user_id="u")
        with pytest.raises(am.AtomicMemoryFailure, match="identity"):
            am.insert_atom_vectors(db, prepared, replace(embedding, version="wrong"))
        with pytest.raises(am.AtomicMemoryFailure, match="not_ready"):
            am.insert_atom_vectors(db, prepared, replace(embedding, vectors=[]))
        with pytest.raises(am.AtomicMemoryFailure, match="embedding_timeout"):
            am.insert_atom_vectors(db, prepared, replace(embedding, status="failed", error_code="embedding_timeout"))
    assert db.execute("SELECT COUNT(*) FROM atomic_vectors").fetchone()[0] == 0


def test_vectors_savepoint_and_parent_rollback_no_partial_publication(db):
    prepared = publish(db, [raw(), raw("r2")])
    embedding = vector_batch(prepared)
    second_id = sorted(a.id for a in prepared.atoms)[1]
    db.execute("BEGIN")
    db.execute(f"""CREATE TRIGGER fail_vector BEFORE INSERT ON atomic_vectors
        WHEN NEW.atom_id='{second_id}' BEGIN SELECT RAISE(ABORT,'vector injected'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="vector injected"):
        am.insert_atom_vectors(db, prepared, embedding)
    assert db.in_transaction and db.execute("SELECT COUNT(*) FROM atomic_vectors").fetchone()[0] == 0
    db.rollback()
    db.execute("BEGIN")
    am.insert_atom_vectors(db, prepared, embedding)
    assert db.execute("SELECT COUNT(*) FROM atomic_vectors").fetchone()[0] == 2
    db.rollback()
    assert db.execute("SELECT COUNT(*) FROM atomic_vectors").fetchone()[0] == 0


@pytest.mark.parametrize("mutation", ["text_hash", "binding_hash", "metadata", "zero", "boolean", "nonfinite"])
def test_persisted_vector_corruption_not_searchable_or_ready_and_pruned(db, mutation):
    prepared = publish(db, [raw()])
    embedding = vector_batch(prepared, [[1.0, 0.0]])
    with transaction(db):
        am.insert_atom_vectors(db, prepared, embedding)
        sql = {"text_hash": "UPDATE atomic_vectors SET atom_text_hash='bad'",
            "binding_hash": "UPDATE atomic_vectors SET source_binding_hash='bad'",
            "metadata": "UPDATE atomic_vectors SET metadata_json='{}'",
            "zero": "UPDATE atomic_vectors SET vector_json='[0,0]'",
            "boolean": "UPDATE atomic_vectors SET vector_json='[true,1]'",
            "nonfinite": "UPDATE atomic_vectors SET vector_json='[NaN,1]'"}[mutation]
        db.execute(sql)
    result = dense_search(db, prepared, embedding)
    assert not result.candidates and result.trace["invalid_vector_ids"]
    assert am.counts(db, "u", prepared.extractor_version, embedding_version=embedding.version, dimensions=2)["dense"] == 0
    with transaction(db):
        am.prune(db, "u")
    assert db.execute("SELECT COUNT(*) FROM atomic_vectors").fetchone()[0] == 0


@pytest.mark.parametrize("operation", ["delete", "invalidate", "source_change", "raw_delete"])
def test_vector_lifecycle_cleanup_and_late_replay_rejection(db, operation):
    prepared = publish(db, [raw()])
    embedding = vector_batch(prepared, [[1.0, 0.0]])
    with transaction(db):
        am.insert_atom_vectors(db, prepared, embedding)
    with transaction(db):
        if operation == "delete": am.delete_atoms(db, "u", ["r1"])
        if operation == "invalidate": am.invalidate_atoms(db, "u", ["r1"])
        if operation == "source_change": db.execute("UPDATE raw_messages SET content='changed content'")
        if operation == "raw_delete": db.execute("DELETE FROM raw_messages")
        am.prune(db, "u")
    assert db.execute("SELECT COUNT(*) FROM atomic_vectors").fetchone()[0] == 0
    assert dense_search(db, prepared, embedding).candidates == []
    with transaction(db):
        with pytest.raises(am.AtomicMemoryFailure):
            am.insert_atom_vectors(db, prepared, embedding)


@pytest.mark.parametrize("query", [[0.0, 0.0], [True, 1.0], [1.0], [float("nan"), 0.0]])
def test_dense_invalid_query_rejected_before_sql(db, query):
    with pytest.raises(am.AtomicMemoryFailure, match="atom_query_"):
        am.search_atom_dense(db, "u", query, "embedding", 2, "extractor")
    assert not db.in_transaction


def test_vector_revision_switch_is_explicit_and_replay_cannot_rebind(db):
    prepared = publish(db, [raw()])
    first = vector_batch(prepared, [[1.0, 0.0]])
    second = vector_batch(prepared, [[0.0, 1.0]], revision="2")
    with transaction(db):
        am.insert_atom_vectors(db, prepared, first)
        am.insert_atom_vectors(db, prepared, second)
        with pytest.raises(am.AtomicMemoryFailure, match="atom_vector_replay_conflict"):
            am.insert_atom_vectors(db, prepared, replace(first, vectors=[[0.0, 1.0]]))
    assert len(dense_search(db, prepared, first).candidates) == 1
    assert dense_search(db, prepared, second).candidates == []
    assert len(dense_search(db, prepared, second, [0.0, 1.0]).candidates) == 1


def test_invalid_dense_trace_shares_one_candidate_cap(db):
    messages = [raw(f"r{i}") for i in range(5)]
    prepared = publish(db, messages)
    embedding = vector_batch(prepared, [[1.0, 0.0]] * 5)
    with transaction(db):
        am.insert_atom_vectors(db, prepared, embedding)
        db.execute("UPDATE atomic_vectors SET vector_json='[0,0]'")
    result = dense_search(db, prepared, embedding, candidate_limit=2)
    assert result.trace["invalid_count"] == 5
    assert len(result.trace["invalid_vector_ids"]) + len(result.trace["matched_atoms"]) <= 2


def test_bad_credentials_and_invalid_unicode_preflight_never_send(monkeypatch):
    calls = []
    provider = http_provider(monkeypatch, lambda request: calls.append(request))
    monkeypatch.setenv("TEST_ATOM_KEY", "secret with spaces")
    assert am.prepare_atoms([raw()], provider, before_dispatch=lambda: None).error_code == "atom_invalid_api_key"
    assert not calls
    prepared = am.prepare_atoms([raw(content="unpaired\ud800")], StubProvider())
    assert prepared.error_code == "atom_invalid_source_unicode"


@pytest.mark.parametrize("policy_sql", ["retraction_state='retracted'", "disclosure_policy='restricted'",
    "disclosure_policy='blocked'", "visibility='suppressed'"])
def test_policy_only_version_rebind_preserves_completion_and_vectors_without_disclosure(db, policy_sql):
    prepared = publish(db, [raw()])
    embedding = vector_batch(prepared, [[1.0, 0.0]])
    original_atom = prepared.atoms[0]
    with transaction(db):
        am.insert_atom_vectors(db, prepared, embedding)
        db.execute(f"UPDATE source_policies SET {policy_sql},source_version=2")
        assert am.prune(db, "u") == 0
        assert am.counts(db, "u", prepared.extractor_version, embedding_version=embedding.version, dimensions=2)["dense"] == 1
    new_atom = am.Atom.model_validate_json(db.execute("SELECT payload_json FROM atomic_memories").fetchone()[0])
    assert new_atom.id != original_atom.id and new_atom.source_version == 2
    assert new_atom.draft == original_atom.draft
    assert db.execute("SELECT COUNT(*) FROM atomic_vectors").fetchone()[0] == 1
    assert search(db, prepared).candidates == dense_search(db, prepared, embedding).candidates == []
    assert am.counts(db, "u", prepared.extractor_version)["atom"] == 1
    if "restricted" in policy_sql:
        assert len(search(db, prepared, allowed_restricted_ids=frozenset({"r1"})).candidates) == 1
    if "retracted" in policy_sql:
        assert len(search(db, prepared, historical=True).candidates) == 1
    with transaction(db):
        db.execute("UPDATE source_policies SET retraction_state='not_retracted',disclosure_policy='within_user',visibility='visible',source_version=3")
        assert am.prune(db, "u") == 0
    assert len(search(db, prepared).candidates) == len(dense_search(db, prepared, embedding).candidates) == 1
    restored = am.Atom.model_validate_json(db.execute("SELECT payload_json FROM atomic_memories").fetchone()[0])
    assert restored.source_version == 3 and restored.draft == original_atom.draft
    assert am.counts(db, "u", prepared.extractor_version, embedding_version=embedding.version, dimensions=2)["dense"] == 1


def test_zero_atom_receipt_rebind_requires_no_model_or_vectors(db):
    provider = StubProvider([])
    prepared = publish(db, [raw()], provider)
    embedding = vector_batch(prepared, [])
    with transaction(db):
        db.execute("UPDATE source_policies SET retraction_state='retracted',source_version=2")
        assert am.rebind_versions(db, "u") == 0
        am.prune(db, "u")
    assert len(provider.calls) == 1
    assert db.execute("SELECT source_version FROM atomic_extractions").fetchone()[0] == 2
    count = am.counts(db, "u", prepared.extractor_version, embedding_version=embedding.version, dimensions=2)
    assert count["active"] == count["atom"] == count["fts"] == count["dense"] == 1
    assert not search(db, prepared).candidates


def test_linked_retraction_rebinds_all_dependency_receipts_and_vector(db):
    first, second = raw(), raw("r2", "context")
    prepared = publish(db, [first, second], StubProvider([draft(first, text="east", source_refs=[ref(first), ref(second)])]))
    embedding = vector_batch(prepared)
    with transaction(db):
        am.insert_atom_vectors(db, prepared, embedding)
        db.execute("UPDATE source_policies SET retraction_state='retracted',source_version=2 WHERE raw_id='r2'")
        assert am.rebind_versions(db, "u") == 1
        am.prune(db, "u")
    assert am.counts(db, "u", prepared.extractor_version, embedding_version=embedding.version, dimensions=2)["dense"] == 2
    assert search(db, prepared, "east").candidates == []
    assert {h.id for h in search(db, prepared, "east", historical=True).candidates} == {"r1", "r2"}
    assert {r[0]: r[1] for r in db.execute("SELECT source_raw_id,source_version FROM atomic_extractions")} == {"r1": 1, "r2": 2}


def test_rebind_is_transactional_and_never_crosses_invalidation_or_changed_body(db):
    prepared = publish(db, [raw()])
    original = db.execute("SELECT id FROM atomic_memories").fetchone()[0]
    db.execute("BEGIN")
    db.execute("UPDATE source_policies SET retraction_state='retracted',source_version=2")
    assert am.rebind_versions(db, "u") == 1
    db.rollback()
    assert db.execute("SELECT id FROM atomic_memories").fetchone()[0] == original
    with transaction(db):
        am.invalidate_atoms(db, "u", ["r1"])
        db.execute("UPDATE source_policies SET source_version=2")
        assert am.rebind_versions(db, "u") == 0
        am.prune(db, "u")
    assert db.execute("SELECT COUNT(*) FROM atomic_memories").fetchone()[0] == 0


def test_missing_policy_is_active_denominator_but_never_ready(db):
    prepared = publish(db, [raw()])
    with transaction(db):
        db.execute("DELETE FROM source_policies")
    count = am.counts(db, "u", prepared.extractor_version)
    assert count["active"] == 1 and count["raw"] == count["atom"] == 0


def test_deleted_multisource_dependency_preserves_other_active_raw_completion(db):
    first, second = raw(), raw("r2", "context")
    prepared = publish(db, [first, second], StubProvider([draft(first, text="east", source_refs=[ref(first), ref(second)])]))
    embedding = vector_batch(prepared)
    with transaction(db):
        am.insert_atom_vectors(db, prepared, embedding)
        db.execute("UPDATE raw_messages SET content='',deletion_state='deleted' WHERE id='r2'")
        assert am.prune(db, "u") == 1
    count = am.counts(db, "u", prepared.extractor_version, embedding_version=embedding.version, dimensions=2)
    assert count["active"] == count["atom"] == count["fts"] == count["dense"] == 1
    assert count["atoms"] == 0
    assert db.execute("SELECT source_raw_id FROM atomic_extractions").fetchone()[0] == "r1"
    assert db.execute("SELECT COUNT(*) FROM atomic_vectors").fetchone()[0] == 0


@pytest.mark.parametrize("display", [False, True])
def test_parent_raw_profile_display_and_retract_coverage_remains_ready(tmp_path, display):
    from fastapi.testclient import TestClient
    from app.api import create_app
    from app.config import Settings

    settings = Settings(db_path=tmp_path / "lifecycle.sqlite", profile_id="cycle2_raw", enable_extractor=True,
                        atom_endpoint="https://mini.test/chat", enable_governance=True, atom_display=display)
    config = am.AtomicMemoryConfig.from_settings(settings)
    class LocalAtoms(StubProvider):
        def extract(self, messages, *, before_dispatch=None):
            if before_dispatch is not None:
                before_dispatch()
            self.calls.append([m.id for m in messages])
            return am.ExtractionBatch({"atoms": [draft(m, text="atomnavigationmarker") for m in messages]})

    provider = LocalAtoms(config=config)
    with TestClient(create_app(settings, atom_provider=provider)) as client:
        message = "I enjoy tea when fasting."
        def add(text, request):
            return client.post("/add", json={"user_id": "u", "session_id": "s", "request_id": request,
                "messages": [{"role": "user", "content": text}]})
        assert add(message, "original").status_code == 200
        response = client.post("/search", json={"user_id": "u", "query": "atomnavigationmarker", "top_k": 20})
        assert response.status_code == 200, response.text
        content = response.json()["data"][0]["content"]
        assert ("Derived navigation atom" in content) == display
        assert content.endswith(message)
        result = add('I retract my statement that "' + message + '"', "retract")
        assert result.status_code == 200, result.text
        assert len(provider.calls) == 1
        # A retraction must not leave the user permanently unable to add/search.
        result = add("New unrelated note.", "next")
        assert result.status_code == 200, result.text
        response = client.post("/search", json={"user_id": "u", "query": "atomnavigationmarker", "top_k": 20})
        assert response.status_code == 200, response.text
        assert all(message not in h["content"] for h in response.json()["data"])
        with sqlite3.connect(settings.db_path) as connection:
            count = am.counts(connection, "u", config.version)
            assert count["active"] == count["atom"] == count["fts"] == 2
