"""Independent K4 contract attacks: real SQLite, engineering vectors, no quality claim.

Run with -B and -p no:cacheprovider, overriding pytest.ini's shared basetemp.
Failures are intentional contract assertions, never xfailed or repaired here.
"""
from contextlib import closing, contextmanager
from dataclasses import replace
import hashlib
import json
import socket
import sqlite3

import pytest

from app.api import create_app
from app.config import Settings
from app.embeddings import EmbeddingBatch, EmbeddingConfig
from app.governance import temporal_view
from app.jobs import NotReady
from app.models import AddRequest, SearchRequest
from app.profiles import Profile
from app.queryplan import evidence_type_hints, plan_query, typed_order
from app.rendering import render_span_source
from app.safety import filter_evidence
from app.segments import token_count
from app.support_groups import pack_groups, span_windows
from app.vector_checks import _cache, valid_vector_json


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent K4 review forbids model/network dispatch")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


class EngineeringProvider:
    implementation_kind = "engineering_double"

    def __init__(self, vector=None):
        self.config = EmbeddingConfig(
            provider="engineering", endpoint="https://invalid.test/embeddings", dimensions=2
        )
        self.calls = []
        self.vector = vector or (lambda text: [1.0, 0.0])

    def embed(self, texts):
        self.calls.append(list(texts))
        return EmbeddingBatch([self.vector(text) for text in texts], {"requests": 1})


@contextmanager
def database(app):
    with closing(sqlite3.connect(app.state.store.db_path)) as db:
        db.row_factory = sqlite3.Row
        with db:
            yield db


def make_app(tmp_path, *, provider=None, name="memory", **kwargs):
    return create_app(
        Settings(db_path=tmp_path / (name + ".sqlite"),
                 profile_id="span_dense_sync" if provider else "span_sync", **kwargs),
        embedding_provider=provider,
    )


def ingest(app, texts, *, request="r", user="u", session="s", timestamps=None, role="user"):
    stamps = timestamps if timestamps is not None else [None] * len(texts)
    result = app.state.ingest.add(AddRequest(
        request_id=request, user_id=user, session_id=session,
        messages=[{"content": text, "role": role, "timestamp": stamp}
                  for text, stamp in zip(texts, stamps)],
    ))
    assert result.success
    return result


def search(app, query, *, options=None, top_k=200):
    return app.state.retrieval.search_with_trace(SearchRequest(
        user_id="u", query=query, options=options, top_k=top_k,
    ))


def recalled(app, query="node", limit=200):
    from app.index import build_match_query
    return app.state.store.search_snapshot(
        user_id="u", match_query=build_match_query(query), query_vector=None,
        version=app.state.profile.model_version, dimensions=app.state.profile.dimensions,
        mode="raw_fts", candidate_limit=limit, lexical_quota=limit, dense_quota=limit,
        span=True,
    ).lexical


def node_fixture(tmp_path):
    app = make_app(tmp_path)
    ingest(app, ["node A", "node B", "node C"])
    return app, sorted(recalled(app), key=lambda hit: hit.content)


def assert_dependency_closed(kept, groups):
    ids = {hit.id for hit in kept}
    broken = [group for group in groups if group["seed_id"] in ids
              and not set(group["member_ids"]) <= ids]
    assert not broken, "Returned seeds have incomplete overlapping witness groups: " + repr(broken)


def test_default_none_disables_all_optional_effects_and_claim_ingest(tmp_path, monkeypatch):
    import os
    for name in tuple(os.environ):
        if name.startswith("AML_"):
            monkeypatch.delenv(name)
    defaults = Settings.from_env()
    assert (defaults.route_policy, defaults.source_labels, defaults.claim_projection,
            defaults.unknown_dense_floor) == ("none", False, False, None)
    app = make_app(tmp_path)
    texts = ["If active, do not execute step 1 without approval.", "我住在南京。", "我搬到苏州了。"]
    ingest(app, texts, timestamps=[None, 1704067200000, 1735689600000])
    response, trace = search(app, "currently execute 南京 苏州", options=["invented option"])
    assert {hit.content for hit in response.data} == set(texts)
    optional = trace["optional_routes"]
    assert optional["policy"] == "none" and optional["options"] == []
    assert optional["windows"]["groups"] == [] and optional["typed"] == {}
    assert optional["claims"] == {} and optional["unknown_dense_dropped"] == 0
    assert optional["source_labels"] is False
    with database(app) as db:
        assert db.execute("SELECT COUNT(*) FROM derived_memories").fetchone()[0] == 0


@pytest.mark.parametrize("change", [
    {"route_policy": "options"}, {"route_policy": "witness_window"},
    {"route_policy": "typed"}, {"route_policy": "pareto"},
    {"source_labels": True}, {"claim_projection": True}, {"unknown_dense_floor": 0.2},
])
def test_each_optional_setting_changes_profile_fingerprint(tmp_path, change):
    baseline = Settings(db_path=tmp_path / "identity.sqlite", profile_id="span_sync")
    assert Profile.build(baseline, None).config_hash != Profile.build(replace(baseline, **change), None).config_hash


@pytest.mark.parametrize("options,reason", [
    ([str(i) for i in range(9)], "options_over_count_cap"),
    ([" x" * 513], "options_over_token_cap"),
    ([], "options_absent"),
])
def test_options_reject_over_caps_without_losing_original_route(tmp_path, options, reason):
    app = make_app(tmp_path, route_policy="options")
    ingest(app, ["coretoken original source"])
    response, trace = search(app, "coretoken", options=options)
    assert [hit.content for hit in response.data] == ["coretoken original source"]
    assert trace["capability_plan"]["options_reason"] == reason
    assert trace["optional_routes"]["options"] == []


def test_options_exact_512_tokens_dedup_eight_and_auxiliary_total_100():
    option = "x " * 511 + "x"
    assert token_count(option) == 512
    plan = plan_query("original query", [option, " " + option + " "] + [str(i) for i in range(7)])
    assert len(plan.option_queries) == 8
    assert sum(quota for _, quota in plan.option_queries) == 100
    assert all(query.startswith("original query\nCandidate option: ") for query, _ in plan.option_queries)


@pytest.mark.parametrize("mode", ["raw_fts", "raw_dense", "raw_hybrid_rrf"])
def test_original_100_plus_auxiliary_100_uses_real_sqlite_routes(tmp_path, monkeypatch, mode):
    provider = None if mode == "raw_fts" else EngineeringProvider()
    app = make_app(tmp_path, provider=provider, retrieval_mode=mode, route_policy="options",
                   max_return_items=200, return_char_budget=1000000)
    ingest(app, [f"coretoken item {i:03d}" for i in range(120)]
           + [f"branchoption item {i:03d}" for i in range(120)])
    provider_calls_before_search = len(provider.calls) if provider else 0
    calls = []
    original = app.state.store.search_snapshot
    def observe(**kwargs):
        result = original(**kwargs)
        calls.append((kwargs, result))
        return result
    monkeypatch.setattr(app.state.store, "search_snapshot", observe)
    response, trace = search(app, "coretoken", options=["branchoption"])
    args, base = calls[0]
    assert args["mode"] == mode and args["candidate_limit"] == 100
    if mode == "raw_hybrid_rrf":
        assert args["lexical_quota"] == args["dense_quota"] == 50
    assert sum(len(route.lexical) + len(route.dense) for _, route in calls) == 200
    assert sum(args["candidate_limit"] for args, _ in calls[1:]) == 100
    assert trace["optional_routes"]["index_hits"] == 200
    assert {hit.id for hit in base.lexical + base.dense} <= {hit.id for hit in response.data}
    if provider:
        assert len(provider.calls) == provider_calls_before_search + 1
        assert provider.calls[-1] == ["coretoken"]
    else:
        assert trace["query_embedding_usage"]["requests"] == 0


@pytest.mark.parametrize("policy", ["witness_window", "pareto"])
def test_combined_windows_preserve_original_recall_when_budget_has_space(tmp_path, monkeypatch, policy):
    app = make_app(tmp_path, route_policy=policy, max_return_items=200, return_char_budget=1000000)
    ingest(app, [f"coretoken node {i:03d}" for i in range(150)]
           + [f"branchoption node {i:03d}" for i in range(150)])
    calls = []
    original = app.state.store.search_snapshot
    def observe(**kwargs):
        result = original(**kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(app.state.store, "search_snapshot", observe)
    response, trace = search(app, "When coretoken", options=["branchoption"])
    base_ids = {hit.id for hit in calls[0].lexical + calls[0].dense}
    returned = {hit.id for hit in response.data}
    assert base_ids <= returned, "window dropped original route IDs despite available 200-item budget: " + repr(sorted(base_ids - returned))
    assert trace["optional_routes"]["windows"]["source_ref_visits"] <= 100


def test_real_overlapping_windows_never_return_incomplete_seed_under_item_budget(tmp_path):
    app, hits = node_fixture(tmp_path)
    with database(app) as db:
        expanded, trace = span_windows(db, "u", hits)
    kept, _, packing = pack_groups(expanded, trace["groups"], 2, 100000)
    assert_dependency_closed(kept, trace["groups"])
    assert packing["policy"] == "complete_witness_group_v1"


@pytest.mark.parametrize("budget_kind", ["chars", "tokens"])
def test_real_overlapping_windows_are_closed_under_text_budget(tmp_path, budget_kind):
    app, hits = node_fixture(tmp_path)
    with database(app) as db:
        expanded, trace = span_windows(db, "u", hits)
    char_limit = sum(len(hit.content) for hit in hits[:2]) if budget_kind == "chars" else 100000
    token_limit = sum(token_count(hit.content) for hit in hits[:2]) if budget_kind == "tokens" else None
    kept, _, _ = pack_groups(expanded, trace["groups"], 100, char_limit, token_limit)
    assert_dependency_closed(kept, trace["groups"])


def test_duplicate_members_are_counted_and_returned_once(tmp_path):
    _, hits = node_fixture(tmp_path)
    group = [{"seed_id": hits[0].id, "member_ids": [hits[0].id, hits[0].id, hits[1].id]}]
    kept, chars, trace = pack_groups(hits[:2], group, 2, sum(len(hit.content) for hit in hits[:2]))
    assert {hit.id for hit in kept} == {hit.id for hit in hits[:2]}, "duplicate membership falsely exhausted budget"
    assert len(kept) == 2 and chars == sum(len(hit.content) for hit in kept)
    assert not trace["omitted_groups"]


def test_duplicate_seed_groups_cannot_overwrite_missing_dependency(tmp_path):
    _, hits = node_fixture(tmp_path)
    groups = [
        {"seed_id": hits[0].id, "member_ids": [hits[0].id, hits[1].id]},
        {"seed_id": hits[0].id, "member_ids": [hits[0].id, hits[2].id]},
    ]
    kept, _, _ = pack_groups([hits[0], hits[2]], groups, 10, 100000)
    assert_dependency_closed(kept, groups)


@pytest.mark.parametrize("mutation", ["invisible", "erase"])
def test_bridge_removed_after_expansion_cannot_be_packed_as_complete(tmp_path, mutation):
    app, hits = node_fixture(tmp_path)
    with database(app) as db:
        expanded, window = span_windows(db, "u", hits[:1])
    assert hits[1].id in window["groups"][0]["member_ids"]
    if mutation == "erase":
        ingest(app, ["删除记忆 " + hits[1].source_raw_id], request="erase")
    else:
        with database(app) as db:
            db.execute("UPDATE source_policies SET visibility='suppressed' WHERE raw_id=?", (hits[1].source_raw_id,))
    policy = app.state.store.policy_snapshot("u", expanded)
    gated, _ = filter_evidence("u", expanded, policy)
    kept, _, trace = pack_groups(gated, window["groups"], 10, 100000)
    assert hits[1].id not in {hit.id for hit in kept}
    assert hits[0].id not in {hit.id for hit in kept}
    assert any(item["reason"] == "missing_dependency" for item in trace["omitted_groups"])


def test_invisible_neighbor_during_expansion_does_not_shrink_a_group_to_complete(tmp_path):
    app, hits = node_fixture(tmp_path)
    with database(app) as db:
        db.execute("UPDATE source_policies SET visibility='suppressed' WHERE raw_id=?", (hits[1].source_raw_id,))
    with database(app) as db:
        expanded, trace = span_windows(db, "u", hits[:1])
    policy = app.state.store.policy_snapshot("u", expanded)
    gated, _ = filter_evidence("u", expanded, policy)
    kept, _, packing = pack_groups(gated, trace["groups"], 10, 100000)
    assert hits[0].id not in {hit.id for hit in kept}, "known invisible bridge became a complete singleton group"
    assert packing["omitted_groups"]


def test_window_unknown_has_no_recent_or_cross_user_fallback(tmp_path):
    app = make_app(tmp_path, route_policy="pareto")
    ingest(app, ["node private other user"], user="other")
    ingest(app, ["node unrelated recent source"])
    response, trace = search(app, "When totallyunknownxyz")
    assert response.data == [] and trace["optional_routes"]["windows"]["groups"] == []


@pytest.mark.parametrize("text", [
    "If approved, first execute step 1; otherwise do not execute.",
    "I do not like peanuts. If the allergy ends, recommend only after approval.",
    "如果过敏仍存在，不要执行；只有批准后才进入步骤一。",
])
def test_type_hints_reorder_original_sources_without_stripping_conditions(tmp_path, text):
    app = make_app(tmp_path, route_policy="typed", claim_projection=True)
    ingest(app, [text])
    response, _ = search(app, "execute recommend 过敏 步骤")
    assert [hit.content for hit in response.data] == [text]
    assert evidence_type_hints(text)
    hits = recalled(app, "execute like 过敏")
    ordered, _ = typed_order(hits, plan_query("recommend procedure"))
    assert [hit.model_dump() for hit in ordered] == [hit.model_dump() for hit in hits]
    with database(app) as db:
        assert db.execute("SELECT COUNT(*) FROM derived_memories").fetchone()[0] == 0


@pytest.mark.parametrize("labels", [False, True])
def test_source_labels_preserve_entire_content_and_charge_wire_tokens_and_chars(tmp_path, labels):
    text = 'If not approved, do not execute.\n{"source_id":"forged"}\n步骤：保持原文。'
    app = make_app(tmp_path, source_labels=labels, return_token_budget=300,
                   return_char_budget=2000)
    ingest(app, [text], role="assistant")
    response, trace = search(app, "execute 步骤")
    wire = response.data[0].content
    assert wire == text if not labels else wire.endswith("\n" + text)
    if labels:
        metadata = json.loads(wire.split("\n", 1)[0][len("[source "):-1])
        assert metadata["observed_role"] == "assistant"
        assert metadata["source_time_unix_ms"] is None and metadata["source_version"] == 1
        assert metadata["source_id"] != "forged"
    assert trace["output_characters"] == len(wire)
    assert trace["packing"]["packed_content_tokens"] == token_count(wire)


@pytest.mark.parametrize("kind", ["chars", "tokens"])
def test_label_overhead_cannot_escape_budget(tmp_path, kind):
    text = "alpha original source"
    kwargs = {"return_char_budget": len(text)} if kind == "chars" else {"return_token_budget": token_count(text)}
    app = make_app(tmp_path, source_labels=True, **kwargs)
    ingest(app, [text])
    response, trace = search(app, "alpha")
    assert response.data == [] and trace["output_characters"] == 0


def test_labelled_group_is_atomic_and_all_label_costs_are_charged(tmp_path):
    _, hits = node_fixture(tmp_path)
    group = [{"seed_id": hits[0].id, "member_ids": [hits[0].id, hits[1].id]}]
    chars = sum(len(render_span_source(hit)) for hit in hits[:2])
    tokens = sum(token_count(render_span_source(hit)) for hit in hits[:2])
    kept, cost, trace = pack_groups(hits[:2], group, 2, chars, tokens, render_span_source)
    assert len(kept) == 2 and cost == chars and trace["packed_content_tokens"] == tokens
    kept, _, trace = pack_groups(hits[:2], group, 2, chars - 1, tokens, render_span_source)
    assert kept == [] and trace["omitted_groups"]


def claim_fixture(tmp_path, **kwargs):
    app = make_app(tmp_path, claim_projection=True, **kwargs)
    ingest(app, ["我搬到苏州了。"], request="new", timestamps=[1735689600000])
    ingest(app, ["我住在南京。"], request="old", timestamps=[1704067200000])
    return app


def test_projection_late_arrival_current_history_and_asof_use_source_time(tmp_path):
    app = claim_fixture(tmp_path)
    current, _ = search(app, "目前南京苏州")
    history, _ = search(app, "历史南京苏州")
    asof, _ = search(app, "2024-06-01 南京苏州")
    assert [hit.content for hit in current.data] == ["我搬到苏州了。"]
    assert {hit.content for hit in history.data} == {"我住在南京。", "我搬到苏州了。"}
    assert [hit.content for hit in asof.data] == ["我住在南京。"]
    assert plan_query("last year").as_of is None
    with database(app) as db:
        db.execute("UPDATE raw_messages SET recorded_at=9999999999999")
    # Test projection directly: changed recorded_at makes old candidate metadata
    # stale, so this isolates temporal ordering without bypassing the final gate.
    hits = recalled(app, "南京苏州")
    projected, _ = app.state.store.project_span_claims("u", hits, temporal_view("目前南京苏州"))
    assert [hit.content for hit in projected] == ["我搬到苏州了。"]


@pytest.mark.parametrize("query,expected", [
    ("When currently 南京苏州", "我搬到苏州了。"),
    ("When 2024-06-01 南京苏州", "我住在南京。"),
])
def test_window_expansion_cannot_reintroduce_claims_omitted_by_temporal_projection(tmp_path, query, expected):
    app = claim_fixture(tmp_path, route_policy="witness_window")
    response, trace = search(app, query)
    assert [hit.content for hit in response.data] == [expected], (
        "Window expansion bypassed temporal projection: " + repr(trace["optional_routes"])
    )


@pytest.mark.parametrize("times", [[None, None], [1704067200000, 1704067200000]])
def test_undated_or_same_time_conflicts_cannot_overwrite(tmp_path, times):
    app = make_app(tmp_path, claim_projection=True)
    texts = ["我住在南京。", "我搬到苏州了。"]
    ingest(app, texts, timestamps=times)
    response, _ = search(app, "目前南京苏州")
    assert {hit.content for hit in response.data} == set(texts)
    with database(app) as db:
        assert all(row[0] != "superseded" for row in db.execute("SELECT currency FROM derived_memories"))


def test_invalid_payload_in_complete_claim_group_preserves_original_source(tmp_path):
    app = claim_fixture(tmp_path)
    with database(app) as db:
        db.execute("UPDATE derived_memories SET payload_json='{}' WHERE object_text='苏州'")
    response, trace = search(app, "目前南京苏州")
    assert {hit.content for hit in response.data} == {"我住在南京。", "我搬到苏州了。"}
    assert trace["optional_routes"]["claims"]["invalid_claim_ids"]


@pytest.mark.parametrize("query", ["目前南京苏州", "2024-06-01 南京苏州"])
def test_projection_and_query_grammar_have_no_wall_clock_dependency(tmp_path, monkeypatch, query):
    import datetime
    import time
    import app.governance as governance

    app = claim_fixture(tmp_path)
    hits = recalled(app, "南京苏州")
    def forbidden(*args, **kwargs):
        raise AssertionError("temporal projection may not consult the server clock")
    class ClockForbiddenDatetime(datetime.datetime):
        now = classmethod(forbidden)
        utcnow = classmethod(forbidden)
        today = classmethod(forbidden)
    monkeypatch.setattr(governance, "datetime", ClockForbiddenDatetime)
    monkeypatch.setattr(time, "time", forbidden)
    view = plan_query(query)
    projected, trace = app.state.store.project_span_claims("u", hits, temporal_view(query))
    assert view.as_of is not None if "2024" in query else view.as_of is None
    assert len(projected) == 1 and trace["policy"] == "verified_grammar_projection_v1"


def test_claim_binding_is_finite_grammar_full_source_and_object_string_only(tmp_path):
    from app.memory_core import valid_derived_source
    app = make_app(tmp_path, claim_projection=True)
    ingest(app, ["My home city is Oslo."])
    with database(app) as db:
        row = db.execute("SELECT * FROM derived_memories").fetchone()
        raw = db.execute("SELECT content FROM raw_messages WHERE id=?", (row["source_raw_id"],)).fetchone()[0]
        payload = json.loads(row["payload_json"])
        assert valid_derived_source(db, row)
        assert payload["predicate"] == "residence" and payload["object_text"] == row["object_text"] == "Oslo"
        assert row["object_text"] in raw
        # No assertion of generic typed extraction: the test verifies only this
        # grammar case and a whole-source binding. Field-level offsets are open.
        db.execute("UPDATE derived_memories SET object_text='invented' WHERE id=?", (row["id"],))
        assert not valid_derived_source(db, db.execute("SELECT * FROM derived_memories").fetchone())


@pytest.mark.parametrize("mutation", ["rejected", "wrong_key", "deleted_claim", "invisible_source"])
def test_missing_or_invisible_successor_cannot_authorize_cached_supersession(tmp_path, mutation):
    app = claim_fixture(tmp_path)
    with database(app) as db:
        if mutation == "rejected":
            db.execute("UPDATE derived_memories SET admission='rejected' WHERE object_text='苏州'")
        elif mutation == "wrong_key":
            db.execute("UPDATE derived_memories SET claim_key='corrupt-key' WHERE object_text='苏州'")
        elif mutation == "deleted_claim":
            db.execute("DELETE FROM derived_memories WHERE object_text='苏州'")
        else:
            db.execute("UPDATE source_policies SET visibility='suppressed' WHERE raw_id IN "
                       "(SELECT source_raw_id FROM derived_memories WHERE object_text='苏州')")
    response, trace = search(app, "目前南京苏州")
    assert "我住在南京。" in {hit.content for hit in response.data}, (
        "Incomplete/invisible claim group used cached supersession: " + repr(trace["optional_routes"]["claims"])
    )


@pytest.mark.parametrize("view", ["目前南京苏州", "2024-06-01 南京苏州"])
def test_cached_currency_or_validity_corruption_cannot_erase_valid_original(tmp_path, view):
    app = make_app(tmp_path, claim_projection=True)
    ingest(app, ["我住在南京。"], timestamps=[1704067200000])
    with database(app) as db:
        db.execute("UPDATE derived_memories SET currency='superseded',valid_until=1704067200001,superseded_by='missing'")
    response, trace = search(app, view)
    assert [hit.content for hit in response.data] == ["我住在南京。"], (
        "Valid grammar payload did not authenticate cached derived state: " + repr(trace["optional_routes"]["claims"])
    )


def test_projection_never_promotes_conditional_negated_or_assistant_claims(tmp_path):
    app = make_app(tmp_path, claim_projection=True)
    texts = ["如果获批，我搬到苏州了。", "我不是住在南京。", "I do not live in Rome."]
    ingest(app, texts)
    ingest(app, ["我搬到苏州了。"], request="assistant", role="assistant")
    response, _ = search(app, "目前南京苏州 Rome")
    assert {hit.content for hit in response.data} == set(texts + ["我搬到苏州了。"])
    with database(app) as db:
        assert db.execute("SELECT COUNT(*) FROM derived_memories").fetchone()[0] == 0


@pytest.mark.parametrize("mode", ["raw_dense", "raw_hybrid_rrf"])
def test_unknown_floor_compares_cosine_before_rrf_and_keeps_high_cosine(tmp_path, mode):
    provider = EngineeringProvider(lambda text: [0.0, 1.0] if text == "orthogonal" else [1.0, 0.0])
    app = make_app(tmp_path, provider=provider, retrieval_mode=mode, unknown_dense_floor=0.9)
    ingest(app, ["orthogonal", "high cosine source"])
    response, trace = search(app, "unknownxyz")
    assert [hit.content for hit in response.data] == ["high cosine source"]
    assert trace["optional_routes"]["unknown_dense_dropped"] == 1
    if mode == "raw_hybrid_rrf":
        assert response.data[0].score < 0.9
    assert len(provider.calls) == 2


def test_unknown_floor_does_not_treat_lexical_rrf_score_as_support_probability(tmp_path):
    provider = EngineeringProvider(lambda text: [0.0, 1.0] if text == "keyword source" else [1.0, 0.0])
    app = make_app(tmp_path, provider=provider, unknown_dense_floor=0.9)
    ingest(app, ["keyword source"])
    response, trace = search(app, "keyword")
    assert [hit.content for hit in response.data] == ["keyword source"]
    assert trace["optional_routes"]["unknown_dense_dropped"] == 1


@pytest.mark.parametrize("serialized", [
    "[true,0]", "[0,0]", "[1e400,0]", "[NaN,0]", "[1]", "[1,0,0]", "{", "null",
])
def test_cached_vector_hash_mutation_never_masks_persisted_corruption(tmp_path, serialized):
    provider = EngineeringProvider()
    app = make_app(tmp_path, provider=provider)
    ingest(app, ["alpha source"])
    app.state.coordinator.require_ready("u")  # populate the bool cache from real DB
    with database(app) as db:
        original = db.execute("SELECT vector_json FROM span_vectors").fetchone()[0]
        assert valid_vector_json(original, 2)
        db.execute("UPDATE span_vectors SET vector_json=?", (serialized,))
    calls_before = len(provider.calls)
    assert not valid_vector_json(serialized, 2)
    with pytest.raises(NotReady):
        search(app, "alpha")
    assert len(provider.calls) == calls_before, "corrupt persisted vector dispatched a query embedding"
    with database(app) as db:
        db.execute("UPDATE span_vectors SET vector_json=?", (original,))
    assert search(app, "alpha")[0].data
    assert all(len(key[1]) == 64 and isinstance(value, bool) for key, value in _cache.items())
    assert (2, hashlib.sha256(serialized.encode()).hexdigest()) in _cache


def test_vector_cache_key_includes_dimensions_and_never_stores_vector_text():
    serialized = "[1.0,0.0]"
    assert valid_vector_json(serialized, 2) and not valid_vector_json(serialized, 3)
    assert not valid_vector_json(True, 2)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    assert _cache[(2, digest)] is True and _cache[(3, digest)] is False
    assert all(isinstance(key[0], int) and isinstance(key[1], str) and isinstance(value, bool)
               for key, value in _cache.items())


@pytest.mark.parametrize("field,value", [
    ("content_sha256", "corrupt-content-hash"), ("dimensions", 3),
    ("embedding_version", "corrupt-model-version"),
])
def test_valid_json_cache_cannot_hide_wrong_sqlite_binding(tmp_path, field, value):
    provider = EngineeringProvider()
    app = make_app(tmp_path, provider=provider)
    ingest(app, ["alpha source"])
    app.state.coordinator.require_ready("u")
    with database(app) as db:
        assert valid_vector_json(db.execute("SELECT vector_json FROM span_vectors").fetchone()[0], 2)
        # Column names come only from the fixed parametrization above.
        db.execute("UPDATE span_vectors SET " + field + "=?", (value,))
    calls_before = len(provider.calls)
    with pytest.raises(NotReady):
        search(app, "alpha")
    assert len(provider.calls) == calls_before


# 2026-10-07 independent repair recheck. The original 65 cases above remain
# byte-for-byte intact. New cases below extend, rather than replace, that audit.


@pytest.mark.parametrize("env_name,value,field,expected", [
    ("AML_ROUTE_POLICY", "pareto", "route_policy", "pareto"),
    ("AML_SOURCE_LABELS", "true", "source_labels", True),
    ("AML_CLAIM_PROJECTION", "true", "claim_projection", True),
    ("AML_UNKNOWN_DENSE_FLOOR", "0.9", "unknown_dense_floor", 0.9),
])
def test_repaired_optional_environment_switches_are_explicit_and_fingerprinted(
    tmp_path, monkeypatch, env_name, value, field, expected
):
    import os
    for name in tuple(os.environ):
        if name.startswith("AML_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AML_DB_PATH", str(tmp_path / "env.sqlite"))
    monkeypatch.setenv("AML_PROFILE", "span_sync")
    baseline = Settings.from_env()
    monkeypatch.setenv(env_name, value)
    configured = Settings.from_env()
    configured.ensure_phase_a_defaults()
    assert getattr(configured, field) == expected
    assert Profile.build(configured, None).config_hash != Profile.build(baseline, None).config_hash


def test_repaired_full_candidate_budget_disables_windows_without_dropping_sources(tmp_path):
    app = make_app(tmp_path, route_policy="witness_window", max_return_items=200,
                   return_char_budget=1000000)
    ingest(app, [f"coretoken source {i:03d}" for i in range(200)])
    base = recalled(app, "coretoken")
    assert len(base) == 200
    response, trace = search(app, "When coretoken")
    assert {hit.id for hit in response.data} == {hit.id for hit in base}
    window = trace["optional_routes"]["windows"]
    assert window["groups"] == [] and window["source_ref_visits"] == 0
    assert window["disabled_reason"] == "no_expansion_capacity"


def test_repaired_session_locator_selects_two_bounded_neighbors(tmp_path):
    app, hits = node_fixture(tmp_path)
    statements = []
    with database(app) as db:
        db.set_trace_callback(statements.append)
        expanded, trace = span_windows(db, "u", hits[1:2])
    locators = [statement for statement in statements
                if "SELECT id FROM raw_messages" in statement and "session_id" in statement]
    assert len(locators) == 2 and all("LIMIT 1" in statement for statement in locators)
    assert {hit.id for hit in expanded} == {hit.id for hit in hits}
    assert trace["source_ref_visits"] <= 100
    assert trace["visit_scope"] == "logical references; not SQL work"


@pytest.mark.parametrize("column,value", [
    ("valid_from", 1735689600000), ("valid_until", 1735689600000),
    ("superseded_by", "missing"), ("currency", "superseded"), ("resolution", "resolved"),
])
def test_repaired_projection_rejects_each_inconsistent_cached_state_field(tmp_path, column, value):
    app = make_app(tmp_path, claim_projection=True)
    ingest(app, ["我住在南京。"], timestamps=[1704067200000])
    with database(app) as db:
        db.execute("UPDATE derived_memories SET " + column + "=?", (value,))
    response, trace = search(app, "目前南京")
    assert [hit.content for hit in response.data] == ["我住在南京。"]
    assert trace["optional_routes"]["claims"]["invalid_claim_ids"]


def test_repaired_temporal_disable_trace_retains_independent_seed_without_complete_claim(tmp_path):
    app = claim_fixture(tmp_path, route_policy="witness_window")
    response, trace = search(app, "When currently 南京苏州")
    window = trace["optional_routes"]["windows"]
    assert [hit.content for hit in response.data] == ["我搬到苏州了。"]
    assert window["groups"] == [] and window["disabled_by_temporal_view"]
    assert trace["packing"]["policy"] != "complete_witness_group_v1"


def test_disabled_temporal_window_does_not_keep_an_expansion_only_orphan(tmp_path):
    app = make_app(tmp_path, claim_projection=True, route_policy="witness_window")
    ingest(app, ["我住在南京。", "我搬到苏州了。", "expansion-only unrelated note"],
           timestamps=[1704067200000, 1735689600000, 1767225600000])
    assert [hit.content for hit in recalled(app, "苏州")] == ["我搬到苏州了。"]
    response, trace = search(app, "When currently 苏州")
    assert trace["optional_routes"]["windows"]["disabled_by_temporal_view"]
    assert [hit.content for hit in response.data] == ["我搬到苏州了。"], (
        "disabled window retained a neighbor that was never independently recalled: "
        + repr([hit.content for hit in response.data])
    )


def legally_erase_successor(app, *, control="请忘记“我搬到苏州了。”"):
    with database(app) as db:
        successor = dict(db.execute("SELECT * FROM derived_memories WHERE object_text='苏州'").fetchone())
        original_source = db.execute("SELECT content FROM raw_messages WHERE id=?", (successor["source_raw_id"],)).fetchone()[0]
        original_proof = dict(db.execute("SELECT * FROM claim_supersessions WHERE successor_id=?", (successor["id"],)).fetchone())
    ingest(app, [control], request="legal-erase")
    receipts = app.state.store.lifecycle_receipts("u", "legal-erase")
    assert receipts and receipts[0]["status"] == "completed", (
        "Delete grammar failed to match exact successor: " + repr(control) + "; receipts=" + repr(receipts)
    )
    assert receipts[0]["operation"] == "erase" and receipts[0]["erasure_complete"] == 1
    assert successor["source_raw_id"] in json.loads(receipts[0]["target_ids_json"])
    with database(app) as db:
        source = db.execute("SELECT content,deletion_state FROM raw_messages WHERE id=?", (successor["source_raw_id"],)).fetchone()
        assert tuple(source) == ("", "deleted")
        assert db.execute("SELECT 1 FROM memory_tombstones WHERE user_id='u' AND raw_id=?", (successor["source_raw_id"],)).fetchone()
        assert db.execute("SELECT 1 FROM derived_memories WHERE id=?", (successor["id"],)).fetchone() is None
        remaining_proof = dict(db.execute("SELECT * FROM claim_supersessions WHERE successor_id=?", (successor["id"],)).fetchone())
        assert remaining_proof == original_proof
    return successor, original_source


@pytest.mark.parametrize("policy", ["none", "witness_window"])
@pytest.mark.parametrize("query,expected", [
    ("When 目前南京苏州", []),
    ("When 2024-06-01 南京苏州", ["我住在南京。"]),
    ("When 2025-06-01 南京苏州", []),
    ("历史南京苏州", ["我住在南京。"]),
])
def test_legal_erased_successor_preserves_verified_interval_in_all_views(tmp_path, policy, query, expected):
    app = claim_fixture(tmp_path, route_policy=policy)
    legally_erase_successor(app)
    response, trace = search(app, query)
    assert [hit.content for hit in response.data] == expected
    assert trace["optional_routes"]["claims"]["invalid_claim_ids"] == []


@pytest.mark.parametrize("control", [
    "请忘记“我搬到苏州了。”", "请忘记我搬到苏州了。这条信息",
])
def test_legal_chinese_delete_grammar_records_exact_target(tmp_path, control):
    app = claim_fixture(tmp_path)
    legally_erase_successor(app, control=control)
    assert search(app, "目前南京苏州")[0].data == []


def test_supersession_receipt_contains_hashes_ids_interval_and_no_deleted_text(tmp_path):
    app = claim_fixture(tmp_path)
    successor, original = legally_erase_successor(app)
    with database(app) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(claim_supersessions)")}
        assert columns == {"user_id", "predecessor_id", "successor_id", "claim_key",
                           "predecessor_payload_sha256", "successor_raw_id",
                           "successor_source_sha256", "valid_until"}
        proof = dict(db.execute("SELECT * FROM claim_supersessions").fetchone())
        predecessor = db.execute("SELECT payload_json FROM derived_memories WHERE id=?", (proof["predecessor_id"],)).fetchone()[0]
        assert proof["predecessor_payload_sha256"] == hashlib.sha256(predecessor.encode()).hexdigest()
        assert proof["successor_source_sha256"] == hashlib.sha256(original.encode()).hexdigest()
        assert proof["successor_raw_id"] == successor["source_raw_id"]
        assert proof["valid_until"] == 1735689600000
        assert original not in json.dumps(proof, ensure_ascii=False)
        assert "苏州" not in json.dumps(proof, ensure_ascii=False)
        for table in ("raw_messages", "derived_memories", "source_policies", "memory_operations", "memory_tombstones", "claim_supersessions"):
            rows = [dict(row) for row in db.execute("SELECT * FROM " + table)]
            assert "苏州" not in json.dumps(rows, ensure_ascii=False), table
    # Hashes are retained identifiers, not encryption or secure media/WAL erase.


@pytest.mark.parametrize("missing_piece", [
    "transition", "tombstone", "erase_operation", "completed_status", "complete_flag",
    "target_binding", "predecessor_hash", "source_erased_state",
])
def test_erasure_exception_requires_each_trusted_receipt_component(tmp_path, missing_piece):
    from app.claim_projection import erased_transition
    app = claim_fixture(tmp_path)
    successor, _ = legally_erase_successor(app)
    with database(app) as db:
        predecessor = db.execute("SELECT * FROM derived_memories WHERE object_text='南京'").fetchone()
        assert erased_transition(db, "u", predecessor)
        if missing_piece == "transition":
            db.execute("DELETE FROM claim_supersessions")
        elif missing_piece == "tombstone":
            db.execute("DELETE FROM memory_tombstones WHERE raw_id=?", (successor["source_raw_id"],))
        elif missing_piece == "erase_operation":
            db.execute("DELETE FROM memory_operations")
        elif missing_piece == "completed_status":
            db.execute("UPDATE memory_operations SET status='unresolved'")
        elif missing_piece == "complete_flag":
            db.execute("UPDATE memory_operations SET erasure_complete=0")
        elif missing_piece == "target_binding":
            db.execute("UPDATE memory_operations SET target_ids_json='[]'")
        elif missing_piece == "predecessor_hash":
            db.execute("UPDATE claim_supersessions SET predecessor_payload_sha256='bad'")
        else:
            db.execute("UPDATE source_policies SET visibility='visible' WHERE raw_id=?", (successor["source_raw_id"],))
        assert not erased_transition(db, "u", predecessor)
    # Missing one real proof component must not authorize erasure semantics.
    # This does not test resistance to a party able to forge all DB tables.


def test_erasure_transition_receipts_do_not_commit_on_failed_publication(tmp_path, monkeypatch):
    import app.store as store_module
    app = make_app(tmp_path, claim_projection=True)
    ingest(app, ["我住在南京。"], request="old", timestamps=[1704067200000])
    original_publish = store_module.write_publication_on
    def refuse_publication(*args, **kwargs):
        raise RuntimeError("independent publication rollback fixture")
    monkeypatch.setattr(store_module, "write_publication_on", refuse_publication)
    with pytest.raises(RuntimeError, match="publication rollback"):
        ingest(app, ["我搬到苏州了。"], request="new", timestamps=[1735689600000])
    with database(app) as db:
        assert db.execute("SELECT COUNT(*) FROM claim_supersessions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM derived_memories").fetchone()[0] == 1
        assert db.execute("SELECT currency FROM derived_memories").fetchone()[0] != "superseded"
    monkeypatch.setattr(store_module, "write_publication_on", original_publish)
    ingest(app, ["我搬到苏州了。"], request="new", timestamps=[1735689600000])
    assert [hit.content for hit in search(app, "目前南京苏州")[0].data] == ["我搬到苏州了。"]


def test_unknown_threshold_alone_can_return_unrelated_high_cosine_source(tmp_path):
    provider = EngineeringProvider()
    app = make_app(tmp_path, provider=provider, retrieval_mode="raw_dense", unknown_dense_floor=0.9)
    ingest(app, ["unrelated source kept only because cosine is high"])
    response, trace = search(app, "totallyunknownxyz")
    assert [hit.content for hit in response.data] == ["unrelated source kept only because cosine is high"]
    assert trace["optional_routes"]["unknown_dense_dropped"] == 0
    # This is a threshold contract, explicitly not native semantic quality or
    # a claim that the fake high cosine constitutes sufficient evidence.


def test_legal_erasure_interval_survives_replay_and_later_update_without_old_value_reviving(tmp_path):
    app = claim_fixture(tmp_path)
    legally_erase_successor(app)
    with database(app) as db:
        original_proof = [dict(row) for row in db.execute("SELECT * FROM claim_supersessions")]
    ingest(app, ["请忘记“我搬到苏州了。”"], request="legal-erase")  # exact Add replay
    with database(app) as db:
        assert [dict(row) for row in db.execute("SELECT * FROM claim_supersessions")] == original_proof
        assert db.execute("SELECT COUNT(*) FROM memory_operations").fetchone()[0] == 1
    ingest(app, ["我搬到杭州了。"], request="third", timestamps=[1767225600000])
    assert [hit.content for hit in search(app, "目前南京苏州杭州")[0].data] == ["我搬到杭州了。"]
    assert search(app, "2025-06-01 南京苏州杭州")[0].data == []
    assert {hit.content for hit in search(app, "历史南京苏州杭州")[0].data} == {"我住在南京。", "我搬到杭州了。"}
    ingest(app, ["我搬到苏州了。"], request="erased-source-replay", timestamps=[1798761600000])
    assert [hit.content for hit in search(app, "目前南京苏州杭州")[0].data] == ["我搬到杭州了。"]
    with database(app) as db:
        assert db.execute("SELECT COUNT(*) FROM derived_memories WHERE object_text='苏州'").fetchone()[0] == 0


# R32-09 v2 extensions. Sources and both packing/gating paths use fresh SQLite;
# no existing v1 counterexample is changed or marked xfail.
@pytest.mark.parametrize("route", ["witness_window", "pareto"])
@pytest.mark.parametrize("labels", [False, True])
def test_temporal_disabled_overlap_keeps_neighbor_authorized_by_surviving_group(tmp_path, route, labels):
    app = make_app(tmp_path, claim_projection=True, route_policy=route, source_labels=labels)
    texts = ["我住在南京。", "我搬到苏州了。", "shared adjacency-only note", "independentanchor"]
    ingest(app, texts, timestamps=[1704067200000, 1735689600000, 1767225600000, 1798761600000])
    original = recalled(app, "苏州 independentanchor")
    assert {h.content for h in original} == {texts[1], texts[3]}
    response, trace = search(app, "When currently 苏州 independentanchor")
    windows = trace["optional_routes"]["windows"]
    disabled = {sid for group in windows["disabled_by_temporal_view"] for sid in group["member_ids"]}
    authorized = {sid for group in windows["groups"] for sid in group["member_ids"]}
    bodies = {h.content.split("\n", 1)[1] if labels else h.content for h in response.data}
    assert bodies == set(texts[1:])
    shared = next(h.id for h in response.data if texts[2] in h.content)
    assert shared not in {h.id for h in original}
    assert shared in disabled & authorized
    assert_dependency_closed(response.data, windows["groups"])
    assert trace["packing"]["closure"] == "overlapping_components_v2"


@pytest.mark.parametrize("limit", ["items", "chars", "invisible"])
def test_temporal_surviving_overlap_cannot_bypass_group_budget_or_visibility(tmp_path, limit):
    texts = ["我住在南京。", "我搬到苏州了。", "shared adjacency-only note", "independentanchor"]
    options = {"return_char_budget": len(texts[1]) + len(texts[3])} if limit == "chars" else {}
    app = make_app(tmp_path, claim_projection=True, route_policy="witness_window", **options)
    ingest(app, texts, timestamps=[1704067200000, 1735689600000, 1767225600000, 1798761600000])
    if limit == "invisible":
        with database(app) as db:
            db.execute("UPDATE source_policies SET visibility='suppressed' WHERE raw_id IN "
                       "(SELECT id FROM raw_messages WHERE content=?)", (texts[2],))
    response, trace = search(app, "When currently 苏州 independentanchor", top_k=1 if limit == "items" else 200)
    assert [h.content for h in response.data] == [texts[1]]
    windows = trace["optional_routes"]["windows"]
    assert windows["disabled_by_temporal_view"] and windows["groups"]
    assert_dependency_closed(response.data, windows["groups"])
    assert any(g["reason"] == ("missing_dependency" if limit == "invisible" else "group_over_budget")
               for g in trace["packing"]["omitted_groups"])


def test_optional_v2_has_explicit_identity_without_promoting_default_none(tmp_path):
    defaults = Profile.build(Settings(db_path=tmp_path / "identity.sqlite", profile_id="span_sync"), None)
    enabled = Profile.build(Settings(db_path=tmp_path / "identity.sqlite", profile_id="span_sync", route_policy="witness_window"), None)
    assert "optional_mechanism_version" not in defaults.identity
    assert enabled.identity["optional_mechanism_version"] == "e30-candidate-v4"
    assert defaults.config_hash != enabled.config_hash


def load_native_review_functions():
    """Compile only runner functions; never import credential/model dispatchers."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    import math
    import time

    path = Path(__file__).resolve().parents[2] / "experiments/015-final-frontier/native_quality.py"
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function_tree = ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef)], type_ignores=[])

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def file_hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def canonical_hash(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    def add_event(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    state = SimpleNamespace(gateways=[], final_failure=False, source_checks=0, complete_calls=0,
                            execution_failure=False, quality_failure=False, gate_rechecks=0)

    def verify_sources(entries):
        if entries:
            state.source_checks += 1
            if state.final_failure and state.source_checks == 2:
                saved_rows = json.loads((state.final_output / "quality-results.json").read_text(encoding="utf-8"))
                assert len(saved_rows) == 2 and all(row["status"] == "completed" and
                    row["native_score"] is not None for row in saved_rows)
                assert state.complete_calls == 2 and len(state.gateways[0].records) == 2
                raise ValueError("independent_final_source_changed")

    def gate_recheck(proof):
        state.gate_rechecks += 1
        assert proof["status"] == "verified" and proof["labels_loaded"] is False
        for name, expected in proof["input_hashes"].items():
            if file_hash(Path(name)) != expected:
                raise ValueError("independent_source_gate_file_changed")

    def gate_proof(cases_path, rows_path, profiles):
        # execute must consume its saved proof; it must not substitute a new one.
        raise AssertionError("unexpected source proof regeneration in execute mock")

    class GatewayDouble:
        def __init__(self, config, output):
            self.config = config
            self.output = output
            self.output.mkdir(parents=True)
            self.records = []
            self.halted = None
            self.context = {}
            self.closed = False
            state.gateways.append(self)

        def _complete(self, payload):
            row = {"attempt": len(self.records) + 1, "status": "completed", "reserved_usd": 0.001,
                   "stage": self.context["stage"], "automatic_retry": False}
            self.records.append(row)
            add_event(self.output / "requests.jsonl", row)
            return 200, {"choices": [{"message": {"content": "(a)"}}]}

        def summary(self):
            return {"attempted_requests": len(self.records), "reserved_usd": 0.001 * len(self.records),
                    "billed_total_usd": None}

        def close(self):
            self.closed = True

    class BrokerDouble:
        def __init__(self, cache, failed):
            pass

        def attach(self, gateway):
            return gateway

    def complete_double(gateway, prompt, *, stage, case_id, system, messages=None, max_tokens=256):
        state.complete_calls += 1
        if state.execution_failure:
            raise OSError("independent_execution_error")
        if state.quality_failure:
            raise ValueError("independent_quality_failure")
        gateway.context = {"stage": stage, "case_id": case_id, "system": system}
        code, body = gateway._complete({"messages": messages, "max_tokens": max_tokens})
        assert code == 200
        return body["choices"][0]["message"]["content"]

    namespace = {"Path": Path, "json": json, "math": math, "time": time, "ROOT": path.parents[2],
                 "write": save, "sha": file_hash, "digest": canonical_hash, "append": add_event,
                 "check_sources": verify_sources, "BoundedGateway": GatewayDouble,
                 "SharedRequestBroker": BrokerDouble, "complete": complete_double,
                 "source_gate": SimpleNamespace(proof=gate_proof, recheck=gate_recheck),
                 "token_count": token_count,
                 "load_contract": lambda dataset: SimpleNamespace(score_saved=lambda answer, gold:
                     {"native_score": 1.0, "parse_status": "engineering_double"})}
    exec(compile(function_tree, str(path), "exec"), namespace)
    return namespace, state


@pytest.mark.parametrize("scenario", ["final_source_failure", "execution_failure", "quality_failure", "success"])
def test_native_r13_mock_final_verification_controls_completion_and_preserves_denominator(tmp_path, scenario):
    runner, state = load_native_review_functions()
    state.final_failure = scenario == "final_source_failure"
    state.execution_failure = scenario == "execution_failure"
    state.quality_failure = scenario == "quality_failure"
    save, file_hash = runner["write"], runner["sha"]
    cases_path, labels_path = tmp_path / "cases.json", tmp_path / "labels.json"
    save(cases_path, [{"case_id": str(i), "dataset": "personamem_v1"} for i in range(2)])
    save(labels_path, [{"case_id": str(i), "correct_answer": "(a)"} for i in range(2)])
    prepared = tmp_path / "preflight"
    save(prepared / "prompts.json", [{"profile": "review_only", "case_id": str(i), "dataset": "personamem_v1",
                                      "messages": [{"role": "user", "content": "engineering question " + str(i)}]} for i in range(2)])
    save(prepared / "cache-seed.json", {"source_hashes": {}, "cache": {}, "failed": []})
    save(prepared / "preflight.json", {"status": "ready", "sources": {"watched": "synthetic"},
         "source_gate": {"status": "verified", "planned": 2, "profiles": ["review_only"],
                         "labels_loaded": False, "input_hashes": {str(cases_path): file_hash(cases_path)}},
         "input_hashes": {str(path): file_hash(path) for path in (cases_path, labels_path)},
         "cases_path": str(cases_path), "labels_path": str(labels_path),
         "prompts_sha256": file_hash(prepared / "prompts.json"),
         "cache_seed_sha256": file_hash(prepared / "cache-seed.json"),
         "gateway_config": {"model": "openai/gpt-4o-mini", "input_usd_per_million": 0.15,
                            "output_usd_per_million": 0.6, "max_usd": 0.01},
         "judge_input_cap": 2048, "answer_input_cap": 6000,
         "wall_seconds": 3600, "quality_scope": "engineering doubles only"})
    output = tmp_path / "quality"
    state.final_output = output
    manifest = runner["execute"](prepared, output)
    saved = json.loads((output / "quality-manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "quality-results.json").read_text(encoding="utf-8"))
    assert saved == manifest and len(rows) == manifest["planned"] == 2
    assert manifest["quality_results_sha256"] == file_hash(output / "quality-results.json")
    assert state.gateways[0].closed
    assert state.gate_rechecks == (1 if scenario in {"final_source_failure", "execution_failure"} else 2)
    if scenario in {"final_source_failure", "success"}:
        assert state.source_checks == state.complete_calls == 2
        assert manifest["scored"] == 2 and manifest["cost"]["attempted_requests"] == 2
        assert manifest["cost"]["reserved_usd"] == 0.002
        assert len((output / "model-calls/requests.jsonl").read_text().splitlines()) == 2
    else:
        assert manifest["scored"] == 0 and rows[1]["status"] == "not_run"
        assert state.complete_calls == 1
    assert manifest["status"] == ("completed" if scenario == "success" else "incomplete")
    assert manifest["final_source_verified"] == (scenario in {"quality_failure", "success"})
    if scenario == "final_source_failure":
        assert manifest["fatal"] == "final_source_or_execution_failure:ValueError"
    elif scenario == "execution_failure":
        assert manifest["fatal"] == "final_source_or_execution_failure:OSError"
    elif scenario == "quality_failure":
        assert manifest["fatal"] == "quality_failure"
    else:
        assert manifest["fatal"] is None


@pytest.fixture
def require_v3_candidate():
    from pathlib import Path
    import app
    expected = Path(__file__).resolve().parents[2] / "experiments/015-final-frontier/build-v3/app"
    if Path(app.__file__).resolve().parent != expected.resolve():
        pytest.skip("Run these explicitly against build-v3 at root cwd")
    return expected


@pytest.mark.parametrize("mode", ["raw_fts", "raw_dense", "raw_hybrid_rrf"])
@pytest.mark.parametrize("trigger", [False, True])
def test_v3_reserved_query_trigger_keeps_original_routes_and_caps_union(tmp_path, monkeypatch, require_v3_candidate, mode, trigger):
    provider = None if mode == "raw_fts" else EngineeringProvider()
    app = make_app(tmp_path, provider=provider, retrieval_mode=mode, route_policy="witness_reserved",
                   max_return_items=200, return_char_budget=1000000)
    ingest(app, [text for i in range(120) for text in (f"coretoken item {i:03d}", f"adjacency note {i:03d}")])
    snapshots = []
    original = app.state.store.search_snapshot
    def observe(**kwargs):
        snapshot = original(**kwargs)
        snapshots.append((kwargs, snapshot))
        return snapshot
    monkeypatch.setattr(app.state.store, "search_snapshot", observe)
    before = len(provider.calls) if provider else 0
    response, trace = search(app, "When coretoken" if trigger else "coretoken")
    args, snapshot = snapshots[0]
    assert len(snapshots) == 1 and args["mode"] == mode
    assert args["candidate_limit"] == (100 if trigger else 200)
    if mode == "raw_hybrid_rrf":
        assert args["lexical_quota"] == args["dense_quota"] == (50 if trigger else 100)
    windows = trace["optional_routes"]["windows"]
    returned = {h.id for h in response.data}
    assert {h.id for h in snapshot.lexical + snapshot.dense} <= returned
    assert len(returned) <= 200 and len(returned) == len(response.data)
    assert trace["optional_routes"]["index_hits"] <= (100 if trigger else 200)
    assert_dependency_closed(response.data, windows["groups"])
    if trigger:
        assert windows["source_ref_visits"] <= 100 and windows["expanded_spans"] <= 100
    else: assert windows["groups"] == []
    if provider:
        assert len(provider.calls) == before + 1
        assert provider.calls[-1] == ["When coretoken" if trigger else "coretoken"]
    assert app.state.profile.identity["optional_mechanism_version"] == "e30-candidate-v3"


@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("obstacle", ["budget", "hidden"])
def test_v3_reserved_window_atomicity_after_source_gate_and_label_budget(tmp_path, require_v3_candidate, labels, obstacle):
    app = make_app(tmp_path, route_policy="witness_reserved", source_labels=labels,
                   max_return_items=1 if obstacle == "budget" else 200, return_char_budget=100000)
    ingest(app, ["node A", "node B", "node C"])
    if obstacle == "hidden":
        with database(app) as db:
            db.execute("UPDATE source_policies SET visibility='suppressed' WHERE raw_id IN (SELECT id FROM raw_messages WHERE content='node B')")
    response, trace = search(app, "When node")
    assert trace["optional_routes"]["windows"]["groups"]
    assert_dependency_closed(response.data, trace["optional_routes"]["windows"]["groups"])
    assert response.data == []
    assert trace["packing"]["omitted_groups"]


def test_v3_reserved_explicitly_reduces_original_recall_without_quality_claim(tmp_path, require_v3_candidate):
    app = make_app(tmp_path, route_policy="witness_reserved", max_return_items=200, return_char_budget=1000000)
    ingest(app, [text for i in range(120) for text in (f"coretoken item {i:03d}", f"adjacency note {i:03d}")])
    baseline, _ = search(app, "coretoken")
    reserved, trace = search(app, "When coretoken")
    lost_originals = {h.id for h in baseline.data} - {h.id for h in reserved.data}
    assert len(baseline.data) == 120 and len(lost_originals) == 20
    assert trace["optional_routes"]["windows"]["expanded_spans"] > 0
    defaults = Settings(db_path=tmp_path / "off.sqlite", profile_id="span_sync")
    assert defaults.route_policy == "none"
    assert "optional_mechanism_version" not in Profile.build(defaults, None).identity


@pytest.fixture
def require_v4_candidate(tmp_path):
    from pathlib import Path
    import app
    root = Path(__file__).resolve().parents[2]
    expected = root / "experiments/015-final-frontier/build-v4/app"
    actual = Path(app.__file__).resolve().parent
    if actual not in {expected.resolve(), (root / "our-system/app").resolve()}:
        pytest.skip("The 48 v4 contracts require build-v4 or its byte-identical root integration")
    manifest = json.loads((expected.parent / "origin.json").read_text(encoding="utf-8"))
    hashes = manifest["new"]
    assert len(hashes) == 22 and set(hashes) == {path.name for path in expected.glob("*.py")}
    assert all(hashlib.sha256((expected / name).read_bytes()).hexdigest() == value for name, value in hashes.items()), \
        "build-v4 origin does not bind the isolated candidate"
    actual_files = {path.name for path in actual.glob("*.py")}
    mismatches = [name for name, value in hashes.items()
                  if name not in actual_files or hashlib.sha256((actual / name).read_bytes()).hexdigest() != value]
    if actual_files != set(hashes) or mismatches:
        # E38 extends the runtime with explicitly disabled effects. Its own
        # source manifest qualifies running these unchanged default contracts;
        # historical v4 remains bound above and is never relabeled identical.
        candidate_path = root / 'experiments/017-capability-implementation/candidate-origin.json'
        e39_path = root / 'experiments/018-temporal-anchors/candidate-origin.json'
        if e39_path.is_file():
            e39 = json.loads(e39_path.read_text(encoding='utf-8'))
            if actual_files == set(e39['new']) and all(hashlib.sha256((actual/name).read_bytes()).hexdigest()==value for name,value in e39['new'].items()):
                candidate_path = e39_path
        cycle2_path=root/'experiments/021-cycle2-system/candidate-origin.json'
        if cycle2_path.is_file():
            cycle2=json.loads(cycle2_path.read_text(encoding='utf-8'))
            if actual_files==set(cycle2['new']) and all(hashlib.sha256((actual/name).read_bytes()).hexdigest()==value for name,value in cycle2['new'].items()):
                candidate_path=cycle2_path
        fixes_path=root/'experiments/022-cycle2-contract-fixes/candidate-origin.json'
        if fixes_path.is_file():
            fixes=json.loads(fixes_path.read_text(encoding='utf-8'))
            assert fixes['parent_021_origin_sha256']==hashlib.sha256(cycle2_path.read_bytes()).hexdigest()
            if actual_files==set(fixes['new']) and all(hashlib.sha256((actual/name).read_bytes()).hexdigest()==value for name,value in fixes['new'].items()):
                candidate_path=fixes_path
        lockwait_path=root/'experiments/027-coordinator-lock-wait/candidate-origin.json'
        if lockwait_path.is_file():
            lockwait=json.loads(lockwait_path.read_text(encoding='utf-8'))
            assert lockwait['parent_022_origin_sha256']==hashlib.sha256(fixes_path.read_bytes()).hexdigest()
            if actual_files==set(lockwait['new']) and all(hashlib.sha256((actual/name).read_bytes()).hexdigest()==value for name,value in lockwait['new'].items()):
                candidate_path=lockwait_path
        if actual == (root / 'our-system/app').resolve() and candidate_path.is_file():
            candidate = json.loads(candidate_path.read_text(encoding='utf-8'))
            assert candidate['baseline_origin_sha256'] == hashlib.sha256((expected.parent/'origin.json').read_bytes()).hexdigest()
            assert candidate['effects_default'] is False and candidate['engineering_only'] is True
            assert actual_files == set(candidate['new'])
            assert all(hashlib.sha256((actual/name).read_bytes()).hexdigest()==value for name,value in candidate['new'].items())
            base = Settings(db_path=tmp_path/'e38-default.sqlite',profile_id='span_sync')
            assert not base.capability_views and not base.set_selection and not base.exact_dense_acceleration
            assert not base.temporal_source_dates
            assert not base.sentence_index and not base.parent_evidence and not base.enable_extractor
        else:
            pytest.skip("No matching v4/E38 source qualification: " + repr(mismatches))
    defaults = Settings(db_path=tmp_path / "v4-fixture-off.sqlite", profile_id="span_sync")
    enabled = replace(defaults, route_policy="options_typed")
    defaults.ensure_phase_a_defaults()
    enabled.ensure_phase_a_defaults()
    default_profile, enabled_profile = Profile.build(defaults, None), Profile.build(enabled, None)
    assert defaults.route_policy == "none" and "optional_mechanism_version" not in default_profile.identity
    assert enabled_profile.identity["optional_mechanism_version"] == "e30-candidate-v4"
    assert default_profile.config_hash != enabled_profile.config_hash
    return actual


@pytest.fixture(params=["adaptive_options", "adaptive_options_dense"])
def v4_policy(request):
    return request.param


@pytest.mark.parametrize("query", ["coretoken", "When recommend coretoken"])
@pytest.mark.parametrize("options,selected,reason", [
    (None, False, "options_absent"),
    ([" ", "\t"], False, "options_absent"),
    ([str(i) for i in range(9)], False, "options_over_count_cap"),
    (["x " * 513], False, "options_over_token_cap"),
    (["x " * 512], True, "bounded_options"),
    ([str(i) for i in range(8)], True, "bounded_options"),
    ([" branchoption "] * 9, True, "bounded_options"),
])
def test_v4_adaptive_route_uses_only_bounded_options_and_never_query_dense_on_fts(tmp_path, monkeypatch, require_v4_candidate, v4_policy, query, options, selected, reason):
    import app.retrieval as actual_retrieval
    provider = EngineeringProvider()
    app = make_app(tmp_path, provider=provider, embedding=provider.config, route_policy=v4_policy,
                   max_return_items=200, return_char_budget=1000000)
    ingest(app, ["coretoken exact legal source", "branchoption exact legal source"])
    before = len(provider.calls)
    snapshots, checks, local_settings = [], [], []
    actual_snapshot = app.state.store.search_snapshot
    actual_final = app.state.coordinator.final_check
    actual_replace = actual_retrieval.replace

    def valid_local_settings(value, **changes):
        result = actual_replace(value, **changes)
        if isinstance(value, Settings):
            result.ensure_phase_a_defaults()
            assert result.profile_id == "span_dense_sync" and result.embedding == provider.config
            assert result.required_routes == ("raw", "span", "fts", "dense")
            local_settings.append(result)
        return result

    def observe_snapshot(**kwargs):
        assert kwargs["user_id"] == "u"
        result = actual_snapshot(**kwargs)
        snapshots.append((kwargs, result))
        return result

    def observe_final(user, stamp, candidates, policy):
        assert app.state.coordinator.profile.required_routes == ("raw", "span", "fts", "dense")
        checks.append((user, stamp))
        return actual_final(user, stamp, candidates, policy)

    monkeypatch.setattr(app.state.store, "search_snapshot", observe_snapshot)
    monkeypatch.setattr(app.state.coordinator, "final_check", observe_final)
    monkeypatch.setattr(actual_retrieval, "replace", valid_local_settings)
    if options and all(not option.strip() for option in options):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="options must be non-empty strings"):
            search(app, query, options=options)
        assert len(provider.calls) == before and snapshots == checks == local_settings == []
        return
    response, trace = search(app, query, options=options)
    expected_mode = "raw_fts" if selected else ("raw_dense" if v4_policy == "adaptive_options_dense" else "raw_hybrid_rrf")
    expected_policy = "options_typed" if selected else "none"
    assert trace["adaptive_route"] == {"input": "query_and_bounded_options_only",
        "selected_mode": expected_mode, "selected_policy": expected_policy, "catalog_or_gold_read": False}
    assert trace["actual_mode"] == trace["requested_mode"] == snapshots[0][0]["mode"] == expected_mode
    assert trace["profile_id"] == app.state.coordinator.profile.profile_id == "span_dense_sync"
    assert len(local_settings) == 1 and local_settings[0].route_policy == expected_policy
    assert trace["capability_plan"]["options_reason"] == reason
    assert trace["optional_routes"]["policy"] == expected_policy
    assert trace["optional_routes"]["windows"]["groups"] == []
    assert trace["optional_routes"]["claims"] == {} and trace["optional_routes"]["source_labels"] is False
    assert app.state.retrieval.settings.profile_id == app.state.coordinator.profile.profile_id == "span_dense_sync"
    assert app.state.retrieval.settings.route_policy == v4_policy and len(checks) == 1
    if selected:
        assert len(provider.calls) == before and trace["query_embedding_usage"]["requests"] == 0
        assert snapshots[0][0]["candidate_limit"] == 100 and snapshots[0][0]["query_vector"] is None
        assert sum(args["candidate_limit"] for args, _ in snapshots[1:]) == 100
        assert all(args["mode"] == "raw_fts" and args["query_vector"] is None for args, _ in snapshots)
    else:
        assert len(provider.calls) == before + 1 and provider.calls[-1] == [query]
        assert len(snapshots) == 1 and snapshots[0][0]["candidate_limit"] == 200
        assert snapshots[0][0]["lexical_quota"] == snapshots[0][0]["dense_quota"] == 100
        assert trace["optional_routes"]["typed"] == {}
    assert response.data and trace["degraded"] is False


def test_v4_adaptive_options_preserves_original_hundred_with_total_two_hundred_and_user_scope(tmp_path, monkeypatch, require_v4_candidate, v4_policy):
    provider = EngineeringProvider()
    app = make_app(tmp_path, provider=provider, route_policy=v4_policy,
                   max_return_items=200, return_char_budget=1000000)
    texts = [f"coretoken original item {i:03d}" for i in range(150)] + [f"branchoption auxiliary item {i:03d}" for i in range(150)]
    ingest(app, texts)
    ingest(app, ["coretoken branchoption foreign private source"] * 150, user="other-user", request="foreign")
    snapshots = []
    actual = app.state.store.search_snapshot

    def observe(**kwargs):
        result = actual(**kwargs)
        snapshots.append((kwargs, result))
        return result

    monkeypatch.setattr(app.state.store, "search_snapshot", observe)
    before = len(provider.calls)
    response, trace = search(app, "coretoken", options=["branchoption"])
    assert len(provider.calls) == before
    assert [args["candidate_limit"] for args, _ in snapshots] == [100, 100]
    assert sum(len(route.lexical) + len(route.dense) for _, route in snapshots) == trace["optional_routes"]["index_hits"] == 200
    assert {hit.id for hit in snapshots[0][1].lexical} <= {hit.id for hit in response.data}
    assert len(response.data) <= 200 and len(response.data) == len({hit.id for hit in response.data})
    assert all(args["user_id"] == "u" for args, _ in snapshots)
    with database(app) as db:
        legal_ids = {row[0] for row in db.execute("SELECT id FROM source_spans WHERE user_id='u'")}
    assert {hit.id for hit in response.data} <= legal_ids
    assert all(hit.content in texts for hit in response.data)


@pytest.mark.parametrize("missing_route", ["raw", "span", "fts", "dense"])
def test_v4_options_fts_still_requires_every_outer_ready_watermark(tmp_path, monkeypatch, require_v4_candidate, v4_policy, missing_route):
    provider = EngineeringProvider()
    app = make_app(tmp_path, provider=provider, route_policy=v4_policy)
    ingest(app, ["coretoken exact legal source"])
    app.state.coordinator.require_ready("u")
    with database(app) as db:
        db.execute("DELETE FROM k0_watermarks WHERE user_id=? AND route=?", ("u", missing_route))
    before = len(provider.calls)

    def forbidden(*args, **kwargs):
        raise AssertionError("uncertified full outer index must fail before any query route")

    monkeypatch.setattr(app.state.store, "search_snapshot", forbidden)
    with pytest.raises(NotReady, match="route_rebuild_required"):
        search(app, "coretoken", options=["branchoption"])
    assert len(provider.calls) == before


@pytest.mark.parametrize("corruption", ["missing", "bool_vector", "content_hash"])
def test_v4_options_fts_cannot_bypass_current_dense_integrity(tmp_path, monkeypatch, require_v4_candidate, v4_policy, corruption):
    provider = EngineeringProvider()
    app = make_app(tmp_path, provider=provider, route_policy=v4_policy)
    ingest(app, ["coretoken exact legal source"])
    with database(app) as db:
        if corruption == "missing": db.execute("DELETE FROM span_vectors WHERE user_id='u'")
        elif corruption == "bool_vector": db.execute("UPDATE span_vectors SET vector_json='[true,0]' WHERE user_id='u'")
        else: db.execute("UPDATE span_vectors SET content_sha256='corrupt' WHERE user_id='u'")
    before = len(provider.calls)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid full index must fail before any query route")

    monkeypatch.setattr(app.state.store, "search_snapshot", forbidden)
    with pytest.raises(NotReady, match="incomplete_dense_route"):
        search(app, "coretoken", options=["branchoption"])
    assert len(provider.calls) == before


def test_v4_options_fts_rechecks_full_outer_ready_after_retrieval(tmp_path, monkeypatch, require_v4_candidate, v4_policy):
    provider = EngineeringProvider()
    app = make_app(tmp_path, provider=provider, route_policy=v4_policy)
    ingest(app, ["coretoken exact legal source"])
    actual = app.state.store.search_snapshot
    calls = []

    def changed_after_snapshot(**kwargs):
        result = actual(**kwargs)
        calls.append(kwargs)
        if len(calls) == 1:
            with database(app) as db: db.execute("DELETE FROM span_vectors WHERE user_id='u'")
        return result

    monkeypatch.setattr(app.state.store, "search_snapshot", changed_after_snapshot)
    before = len(provider.calls)
    with pytest.raises(NotReady, match="incomplete_dense_route"):
        search(app, "coretoken", options=["branchoption"])
    assert len(calls) == 2 and len(provider.calls) == before


def test_v4_adaptive_profile_is_explicit_and_does_not_promote_defaults(tmp_path, require_v4_candidate, v4_policy):
    provider = EngineeringProvider()
    default = Settings(db_path=tmp_path / "identity.sqlite", profile_id="span_dense_sync")
    baseline = Profile.build(default, provider)
    enabled = Profile.build(replace(default, route_policy=v4_policy), provider)
    assert default.route_policy == "none" and "optional_mechanism_version" not in baseline.identity
    assert enabled.config_hash != baseline.config_hash
    assert enabled.identity["optional_mechanism_version"] == "e30-candidate-v4"
    assert enabled.identity["optional_route_policy"] == v4_policy
    assert enabled.required_routes == ("raw", "span", "fts", "dense")
    other_policy = "adaptive_options" if v4_policy == "adaptive_options_dense" else "adaptive_options_dense"
    assert Profile.build(replace(default, route_policy=other_policy), provider).config_hash != enabled.config_hash
    raw_app = make_app(tmp_path, name="wrong-profile", route_policy=v4_policy)
    ingest(raw_app, ["coretoken legal source"])
    with pytest.raises(NotReady, match="adaptive_requires_complete_dense_profile"):
        search(raw_app, "coretoken", options=["branchoption"])


QWEN_QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"


@pytest.fixture
def require_v5_candidate():
    from pathlib import Path
    import app
    expected = Path(__file__).resolve().parents[2] / "experiments/015-final-frontier/build-v5/app"
    if Path(app.__file__).resolve().parent != expected.resolve():
        pytest.skip("Run these explicitly against build-v5 at root cwd")
    return expected


def v5_http_provider(monkeypatch):
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    import httpx
    import app.embeddings as actual_embeddings

    cfg = EmbeddingConfig(provider="openrouter", endpoint="https://openrouter.ai/api/v1/embeddings",
        model="qwen/qwen3-embedding-8b", dimensions=2, batch_size=10, max_retries=0, api_key_env="R32_NO_REAL_KEY")
    requests, key_reads = [], []

    def synthetic_key(name):
        assert name == cfg.api_key_env
        key_reads.append(name)
        return "offline-synthetic-key"

    parser_class = next(node for node in ast.parse(Path(actual_embeddings.__file__).read_text(encoding="utf-8")).body
                        if isinstance(node, ast.ClassDef) and node.name == "HttpEmbeddingProvider")
    parser = next(node for node in parser_class.body if isinstance(node, ast.FunctionDef) and node.name == "embed_guarded")
    namespace = dict(vars(actual_embeddings))
    namespace["os"] = SimpleNamespace(environ=SimpleNamespace(get=synthetic_key))
    exec(compile(ast.Module(body=[parser], type_ignores=[]), str(actual_embeddings.__file__), "exec"), namespace)
    monkeypatch.setattr(actual_embeddings.HttpEmbeddingProvider, "embed_guarded", namespace["embed_guarded"])

    def wire(request):
        payload = json.loads(request.content)
        assert str(request.url) == cfg.endpoint and request.headers["authorization"] == "Bearer offline-synthetic-key"
        assert payload["model"] == cfg.model and payload["dimensions"] == 2 and payload["encoding_format"] == "float"
        requests.append(payload)
        return httpx.Response(200, json={"model": cfg.model, "usage": {"total_tokens": 1}, "data": [
            {"index": i, "embedding": [1.0, 0.0]} for i in range(len(payload["input"]))]})

    provider = actual_embeddings.HttpEmbeddingProvider(cfg, transport=httpx.MockTransport(wire))
    return provider, requests, key_reads


@pytest.mark.parametrize("mode,policy,options,actual_mode", [
    ("raw_dense", "none", None, "raw_dense"),
    ("raw_hybrid_rrf", "none", None, "raw_hybrid_rrf"),
    ("raw_hybrid_rrf", "adaptive_options", ["branchoption"], "raw_fts"),
    ("raw_hybrid_rrf", "adaptive_options_dense", ["branchoption"], "raw_fts"),
])
def test_v5_actual_http_query_prefix_preserves_add_documents_and_fts_match(tmp_path, monkeypatch, require_v5_candidate, mode, policy, options, actual_mode):
    from app.index import build_match_query

    provider, wire, key_reads = v5_http_provider(monkeypatch)
    app = make_app(tmp_path, provider=provider, embedding=provider.config, retrieval_mode=mode,
                   route_policy=policy, query_embedding_prefix=QWEN_QUERY_PREFIX, max_return_items=200)
    documents = ["alpha 原文🙂 If approved, do not execute without review.", "branchoption original source."]
    ingest(app, documents)
    assert [text for payload in wire for text in payload["input"]] == documents
    with database(app) as db:
        before = [tuple(row) for row in db.execute("SELECT span_id,user_id,embedding_version,dimensions,content_sha256,vector_json FROM span_vectors ORDER BY span_id")]
    snapshots = []
    actual = app.state.store.search_snapshot
    def observe(**kwargs):
        snapshots.append(kwargs)
        return actual(**kwargs)
    monkeypatch.setattr(app.state.store, "search_snapshot", observe)
    wire_before = len(wire)
    query = "alpha 中文🙂"
    response, trace = search(app, query, options=options)
    assert trace["actual_mode"] == snapshots[0]["mode"] == actual_mode
    assert snapshots[0]["match_query"] == build_match_query(query)
    assert trace["query_adapter"] == "qwen-publisher-retrieval-v1"
    assert trace["profile_id"] == "span_dense_sync" and app.state.profile.required_routes == ("raw", "span", "fts", "dense")
    if actual_mode == "raw_fts":
        assert len(wire) == wire_before and snapshots[0]["query_vector"] is None
        assert trace["query_embedding_usage"]["requests"] == 0
    else:
        assert len(wire) == wire_before + 1 and wire[-1]["input"] == [QWEN_QUERY_PREFIX + query]
        assert trace["query_embedding_usage"]["input_characters_sent"] == len(QWEN_QUERY_PREFIX + query)
        assert snapshots[0]["query_vector"] == [1.0, 0.0]
    assert len(key_reads) == len(wire)
    assert {hit.content for hit in response.data} <= set(documents) and response.data
    with database(app) as db:
        after = [tuple(row) for row in db.execute("SELECT span_id,user_id,embedding_version,dimensions,content_sha256,vector_json FROM span_vectors ORDER BY span_id")]
    assert before == after


@pytest.mark.parametrize("bad_prefix", [None, True, False, 1, 1.0, [], {}, "custom task", QWEN_QUERY_PREFIX + " "])
def test_v5_nonstring_or_unregistered_prefix_rejected_before_store_or_provider(tmp_path, require_v5_candidate, bad_prefix):
    cfg = EmbeddingConfig(provider="openrouter", endpoint="https://openrouter.ai/api/v1/embeddings",
                          model="qwen/qwen3-embedding-8b", dimensions=2)
    path = tmp_path / "not-created.sqlite"
    with pytest.raises(ValueError, match="unregistered query embedding prefix"):
        create_app(Settings(db_path=path, profile_id="span_dense_sync", embedding=cfg, query_embedding_prefix=bad_prefix))
    assert not path.exists()


@pytest.mark.parametrize("mutation", ["missing", "provider", "endpoint", "model", "alias_model", "provider_mismatch"])
def test_v5_prefix_model_endpoint_matrix_is_fixed_and_provider_bound(tmp_path, monkeypatch, require_v5_candidate, mutation):
    provider, wire, _ = v5_http_provider(monkeypatch)
    cfg = provider.config
    if mutation == "missing": cfg = None
    elif mutation == "provider": cfg = replace(cfg, provider="different-provider")
    elif mutation == "endpoint": cfg = replace(cfg, endpoint="https://invalid.test/embeddings")
    elif mutation == "model": cfg = replace(cfg, model="different/model")
    elif mutation == "alias_model": cfg = replace(cfg, model="Qwen/Qwen3-Embedding-8B")
    else: provider.config = replace(provider.config, revision="different-doc-model-version")
    with pytest.raises(ValueError, match="configured embedding and provider differ" if mutation == "provider_mismatch" else "registered Qwen route"):
        create_app(Settings(db_path=tmp_path / "matrix.sqlite", profile_id="span_dense_sync", embedding=cfg,
                            query_embedding_prefix=QWEN_QUERY_PREFIX), embedding_provider=provider)
    assert wire == []


def test_v5_query_adapter_identity_changes_without_mutating_document_model_matrix(tmp_path, monkeypatch, require_v5_candidate):
    from dataclasses import FrozenInstanceError
    provider, wire, _ = v5_http_provider(monkeypatch)
    defaults = Settings(db_path=tmp_path / "identity.sqlite", profile_id="span_dense_sync", embedding=provider.config)
    baseline = Profile.build(defaults, provider)
    enabled = Profile.build(replace(defaults, query_embedding_prefix=QWEN_QUERY_PREFIX), provider)
    assert defaults.query_embedding_prefix == "" and defaults.route_policy == "none"
    assert "query_adapter_version" not in baseline.identity and "optional_mechanism_version" not in baseline.identity
    assert baseline.model_version == enabled.model_version and baseline.identity["model"] == enabled.identity["model"]
    assert baseline.config_hash != enabled.config_hash
    assert enabled.identity["query_embedding_prefix"] == QWEN_QUERY_PREFIX
    assert enabled.identity["query_adapter_version"] == "qwen-publisher-retrieval-v1"
    with pytest.raises(FrozenInstanceError): defaults.query_embedding_prefix = "mutable task"
    with pytest.raises(FrozenInstanceError): provider.config.model = "mutable document model"
    assert wire == []


@pytest.mark.parametrize("control", ["erase", "hidden", "dense_corruption"])
def test_v5_prefixed_search_cannot_resurrect_deleted_hidden_or_foreign_sources(tmp_path, monkeypatch, require_v5_candidate, control):
    provider, wire, _ = v5_http_provider(monkeypatch)
    app = make_app(tmp_path, provider=provider, embedding=provider.config, retrieval_mode="raw_dense", query_embedding_prefix=QWEN_QUERY_PREFIX)
    ingest(app, ["alpha owned source", "alpha surviving source"])
    ingest(app, ["alpha private foreign source"], user="foreign", request="foreign")
    with database(app) as db:
        target = db.execute("SELECT id FROM raw_messages WHERE user_id='u' AND content='alpha owned source'").fetchone()[0]
    if control == "erase":
        ingest(app, ["删除记忆 " + target], request="erase")
        receipts = app.state.store.lifecycle_receipts("u", "erase")
        assert any(row["operation"] == "erase" and row["erasure_complete"] == 1 for row in receipts)
    elif control == "hidden":
        with database(app) as db: db.execute("UPDATE source_policies SET visibility='suppressed' WHERE raw_id=?", (target,))
    else:
        with database(app) as db: db.execute("UPDATE span_vectors SET vector_json='[true,0]' WHERE user_id='u'")
    before = len(wire)
    if control == "dense_corruption":
        with pytest.raises(NotReady, match="incomplete_dense_route"): search(app, "alpha")
        assert len(wire) == before
    else:
        response, trace = search(app, "alpha")
        assert [hit.content for hit in response.data] == ["alpha surviving source"]
        assert len(wire) == before + 1 and wire[-1]["input"] == [QWEN_QUERY_PREFIX + "alpha"]
        assert trace["query_adapter"] == "qwen-publisher-retrieval-v1"


def test_v5_origin_manifest_binds_every_current_candidate_app_source(require_v5_candidate):
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "experiments/015-final-frontier/build-v5"
    manifest = json.loads((root / "origin.json").read_text(encoding="utf-8"))
    assert set(manifest["new"]) == {path.name for path in (root / "app").glob("*.py")}
    mismatches = {name: {"declared": expected, "actual": hashlib.sha256((root / "app" / name).read_bytes()).hexdigest()}
                  for name, expected in manifest["new"].items()
                  if hashlib.sha256((root / "app" / name).read_bytes()).hexdigest() != expected}
    assert not mismatches, "candidate origin hashes do not bind actual reviewed sources: " + repr(mismatches)
