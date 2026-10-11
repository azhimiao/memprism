"""Cycle2 search contracts for dispatch revocation, shared hit budget, and source occupancy."""
import json
import logging
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import atomic_memory as am
from app import governance
from app.api import create_app
from app.config import Settings
from app.models import AddRequest, EvidenceCandidate, MessageIn, SearchRequest
from app.queryplan import allocate_index_budget, scale_option_queries
from app.retrieval import aggregate_source_occupancy
from app.reranking import EvidenceReranker, RerankConfig
from app.segments import boundaries, token_count
from test_atomic_memory import StubProvider, draft


def payload(texts, request="r", user="u", timestamp=None):
    return dict(user_id=user, session_id="s", request_id=request,
                messages=[dict(role="user", content=text, timestamp=timestamp) for text in texts])


class LocalAtoms(StubProvider):
    def __init__(self, settings, text="needle derived navigation"):
        super().__init__(config=am.AtomicMemoryConfig.from_settings(settings))
        self.text = text

    def extract(self, messages, *, before_dispatch=None):
        if before_dispatch is not None:
            before_dispatch()
        self.calls.append([m.id for m in messages])
        return am.ExtractionBatch({"atoms": [draft(m, text=self.text) for m in messages]})


class Spy:
    def __init__(self):
        self.documents = []

    def rerank(self, query, candidates, *, user_id, groups):
        self.documents.extend(hit.content for hit in candidates)
        return list(candidates), dict(status="spy", count=len(candidates))


def hit(**fields):
    values = dict(id="s", user_id="u", content="abcdef", score=0.2, created_at=1,
                  deletion_state="active", source_raw_id="raw-a", message_index=1,
                  representation="span_exact_v1", span_start=0, span_end=3)
    values.update(fields)
    return EvidenceCandidate(**values)


def test_allocate_keeps_original_share_and_fits_auxiliary_hits():
    combined = allocate_index_budget(200, options_active=True, sentence=True, bridge=True,
                                     atom=False, option_total=100)
    assert combined["original"] == 100
    assert combined["sentence"] > 0 and combined["bridge"] > 0 and combined["options"] > 0
    assert sum(combined.values()) == 200
    options_only = allocate_index_budget(200, options_active=True, sentence=False, bridge=False,
                                          atom=False, option_total=100)
    assert options_only == {"original": 100, "sentence": 0, "bridge": 0, "atom": 0, "options": 100}
    sentence_only = allocate_index_budget(200, options_active=False, sentence=True, bridge=False,
                                           atom=False, option_total=0)
    assert sentence_only["original"] == 150 and sentence_only["sentence"] == 50
    assert sum(sentence_only.values()) == 200
    tiny = allocate_index_budget(3, options_active=True, sentence=True, bridge=True, atom=True, option_total=100)
    assert tiny["original"] >= 1 and sum(tiny.values()) <= 3
    assert scale_option_queries((("q", 60), ("p", 40)), 100) == (("q", 60), ("p", 40))
    assert sum(quota for _, quota in scale_option_queries((("q", 60), ("p", 40)), 50)) == 50


def test_aggregate_merges_covered_spans_and_keeps_distinct_ranges():
    raw = hit(id="raw-a", content="abcdef", representation="raw", span_start=None, span_end=None, score=0.2)
    covered = hit(id="span-1", content="abc", span_start=0, span_end=3, score=0.5)
    other = hit(id="span-b", content="zzz", source_raw_id="raw-b", span_start=0, span_end=3, score=0.1)
    kept, trace = aggregate_source_occupancy([covered, other, raw])
    assert [item.id for item in kept] == ["raw-a", "span-b"]
    assert kept[0].score == pytest.approx(0.7)
    assert trace["dropped_ids"] == ["span-1"]
    left = hit(id="left", content="abc", span_start=0, span_end=3, score=0.4)
    right = hit(id="right", content="def", span_start=3, span_end=6, score=0.3)
    kept, trace = aggregate_source_occupancy([left, right])
    assert [item.id for item in kept] == ["left", "right"] and trace["dropped_ids"] == []
    inside_left = hit(id="left", content="abc", span_start=0, span_end=3, score=0.4)
    inside_right = hit(id="right", content="def", span_start=3, span_end=6, score=0.3)
    kept, trace = aggregate_source_occupancy([inside_left, raw, inside_right])
    assert [item.id for item in kept] == ["raw-a"]
    assert kept[0].score == pytest.approx(0.9)
    assert set(trace["dropped_ids"]) == {"left", "right"}
    outside = hit(id="outside", content="zzz", span_start=0, span_end=3, score=0.4)
    kept, _ = aggregate_source_occupancy([raw, outside])
    assert {item.id for item in kept} == {"raw-a", "outside"}


def test_delete_before_dispatch_reaches_neither_scorer_nor_spy(tmp_path, monkeypatch, caplog):
    documents = []
    original_gate = governance.knowledge_time_gate

    def scorer(query, texts):
        documents.extend(texts)
        return [1.0 for _ in texts]

    settings = Settings(db_path=tmp_path / "dispatch.db", profile_id="cycle2_raw")
    with EvidenceReranker(RerankConfig(enabled=True), scorer=scorer, scorer_id="offline-contract") as ranker:
        app = create_app(settings, reranker=ranker)
        with TestClient(app) as client:
            assert client.post("/add", json=payload(["needle revoked source."])).status_code == 200
            deleted = False

            def interleave(query, candidates, view):
                nonlocal deleted
                if not deleted:
                    deleted = True
                    assert client.post("/add", json=payload(["请忘记“needle revoked source.”"], "delete")).status_code == 200
                return original_gate(query, candidates, view)

            monkeypatch.setattr(governance, "knowledge_time_gate", interleave)
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="aml.retrieval"):
                response = client.post("/search", json=dict(user_id="u", query="needle", top_k=20))
            assert response.status_code == 200 and response.json()["data"] == []
            assert documents == [] and deleted
            records = [record.getMessage() for record in caplog.records if record.name == "aml.retrieval"]
            trace = json.loads(records[-1].removeprefix("retrieval_trace "))
            revocation = trace["dispatch_revocation"][0]
            assert revocation["batches_dispatched"] == 0
            assert revocation["queued_not_sent"] is True
            assert revocation["inflight_recall"] is False

    spy = Spy()
    app = create_app(Settings(db_path=tmp_path / "spy.db", profile_id="cycle2_raw"), reranker=spy)
    with TestClient(app) as client:
        assert client.post("/add", json=payload(["needle revoked source."])).status_code == 200
        deleted = False

        def interleave(query, candidates, view):
            nonlocal deleted
            if not deleted:
                deleted = True
                assert client.post("/add", json=payload(["请忘记“needle revoked source.”"], "delete")).status_code == 200
            return original_gate(query, candidates, view)

        monkeypatch.setattr(governance, "knowledge_time_gate", interleave)
        response = client.post("/search", json=dict(user_id="u", query="needle", top_k=20))
        assert response.status_code == 200, response.text
        assert response.json()["data"] == []
        assert spy.documents == []


def test_live_source_reaches_scorer_when_nothing_is_deleted(tmp_path):
    documents = []

    def scorer(query, texts):
        documents.extend(texts)
        return [1.0 for _ in texts]

    settings = Settings(db_path=tmp_path / "kept.db", profile_id="cycle2_raw")
    with EvidenceReranker(RerankConfig(enabled=True), scorer=scorer, scorer_id="offline-contract") as ranker:
        app = create_app(settings, reranker=ranker)
        with TestClient(app) as client:
            assert client.post("/add", json=payload(["needle kept source."])).status_code == 200
            response = client.post("/search", json=dict(user_id="u", query="needle", top_k=20))
            assert response.status_code == 200 and response.json()["data"]
            assert documents == ["needle kept source."]


def test_dispatched_batch_is_not_recalled_and_queued_batch_is_not_sent(tmp_path):
    calls = []
    state = {"deleted": False, "calls_at_delete": None}
    settings = Settings(db_path=tmp_path / "batches.db", profile_id="cycle2_raw")

    def scorer(query, texts):
        calls.append(list(texts))
        if not state["deleted"]:
            state["deleted"] = True
            forgotten = texts[0]
            added = app.state.ingest.add(AddRequest(
                user_id="u", session_id="s", request_id="delete-one",
                messages=[MessageIn(role="user", content=f"请忘记“{forgotten}”")]))
            assert added.success
            state["forgotten"] = forgotten
            state["calls_at_delete"] = len(calls)
        return [1.0 for _ in texts]

    with EvidenceReranker(RerankConfig(enabled=True, batch_size=1, timeout_seconds=8),
                          scorer=scorer, scorer_id="offline-contract") as ranker:
        app = create_app(settings, reranker=ranker)
        with TestClient(app) as client:
            assert client.post("/add", json=payload(["needle alpha source.", "needle beta source."])).status_code == 200
            response, trace = app.state.retrieval.search_with_trace(
                SearchRequest(user_id="u", query="needle", top_k=20))
    assert state["calls_at_delete"] == 1
    assert len(calls[0]) == 1 and calls[0][0] == state["forgotten"]
    assert all(state["forgotten"] not in text for batch in calls[1:] for text in batch)
    revocation = trace["dispatch_revocation"][0]
    assert revocation["batches_dispatched"] == 1
    assert revocation["batches_cancelled"] >= 1
    assert revocation["queued_not_sent"] is True
    assert revocation["dispatched_results_discarded"] is True
    assert revocation["inflight_recall"] is False
    assert revocation["dispatch_unit"] == "scorer_batch"
    returned = [item.content for item in response.data]
    assert state["forgotten"] not in returned
    assert any(item != state["forgotten"] and "needle" in item for item in returned)


def test_static_source_filters_still_run_before_scoring(tmp_path):
    settings = Settings(db_path=tmp_path / "filter.db", profile_id="cycle2_raw",
                        enable_extractor=True, atom_endpoint="https://mini.test/chat", sentence_index=True,
                        parent_evidence=True, atom_display=True)
    spy = Spy()
    app = create_app(settings, atom_provider=LocalAtoms(settings), reranker=spy)
    with TestClient(app) as client:
        for request, text, user, timestamp in [
            ("allowed", "needle allowed.", "u", 1704067200000),
            ("future", "needle future.", "u", 1735689600000),
            ("blocked", "needle blocked.", "u", 1704067200000),
            ("restricted", "needle restricted.", "u", 1704067200000),
            ("hidden", "needle hidden.", "u", 1704067200000),
            ("retract", "needle retracted.", "u", 1704067200000),
            ("foreign", "needle foreign.", "other", 1704067200000)]:
            assert client.post("/add", json=payload([text], request, user, timestamp)).status_code == 200
        assert client.post("/add", json=payload([
            'I retract my statement that "needle retracted."'], "control")).status_code == 200
        with sqlite3.connect(settings.db_path) as db:
            for name in ("blocked", "restricted"):
                db.execute("UPDATE source_policies SET disclosure_policy=? WHERE raw_id IN "
                           "(SELECT id FROM raw_messages WHERE request_id=?)", (name, name))
            db.execute("UPDATE source_policies SET visibility='suppressed' WHERE raw_id IN "
                       "(SELECT id FROM raw_messages WHERE request_id='hidden')")
        response = client.post("/search", json=dict(
            user_id="u", query="As of 2024-06-01 what was known about needle?", top_k=100))
        assert response.status_code == 200, response.json()
        assert spy.documents == ["needle allowed."]
        assert "Derived navigation atom" in response.json()["data"][0]["content"]
        assert "future" not in response.json()["data"][0]["content"]


@pytest.mark.parametrize("sentence,hops,atom", [
    (False, 0, False), (True, 0, False), (False, 1, False), (True, 1, False),
    (False, 0, True), (True, 1, True),
])
def test_combined_routes_stay_inside_candidate_limit(tmp_path, sentence, hops, atom):
    settings = Settings(db_path=tmp_path / "budget.db", profile_id="cycle2_raw",
                        sentence_index=sentence, bridge_hops=hops, route_policy="options_typed",
                        enable_extractor=atom, atom_endpoint="https://mini.test/chat" if atom else None)
    app = create_app(settings, atom_provider=LocalAtoms(settings) if atom else None)
    with TestClient(app) as client:
        assert client.post("/add", json=payload(
            [f"needle Alice context number {i}." for i in range(120)])).status_code == 200
        response = client.post("/search", json=dict(
            user_id="u", query="What is the relationship needle?", options=["choice"], top_k=100))
        assert response.status_code == 200, response.json()
        _, trace = app.state.retrieval.search_with_trace(SearchRequest(
            user_id="u", query="What is the relationship needle?", options=["choice"], top_k=100))
    budget = trace["optional_routes"]["route_budget"]
    assert budget["unit"] == "index_hit"
    assert budget["planned"]["original"] == min(100, settings.candidate_limit) > 0
    assert budget["planned_total"] <= settings.candidate_limit
    assert budget["actual_total"] <= settings.candidate_limit
    assert budget["actual_total"] == trace["optional_routes"]["index_hits"]
    assert sum(budget["planned"].values()) == budget["planned_total"]
    for name, qty in budget["planned"].items():
        assert budget["actual"][name] <= qty
    if sentence:
        assert budget["planned"]["sentence"] > 0
    if hops:
        assert budget["planned"]["bridge"] > 0
    if atom and sentence:
        assert budget["planned"]["atom"] > 0 and budget["planned"]["options"] > 0


def test_small_candidate_limit_executes_without_budget_failure(tmp_path):
    settings = Settings(db_path=tmp_path / "tiny.db", profile_id="cycle2_raw", candidate_limit=3,
                        sentence_index=True, bridge_hops=1, route_policy="options_typed",
                        enable_extractor=True, atom_endpoint="https://mini.test/chat")
    app = create_app(settings, atom_provider=LocalAtoms(settings))
    with TestClient(app) as client:
        assert client.post("/add", json=payload(
            [f"needle Alice context number {i}." for i in range(8)])).status_code == 200
        response = client.post("/search", json=dict(
            user_id="u", query="What is the relationship needle?", options=["choice"], top_k=20))
        assert response.status_code == 200, response.json()
        _, trace = app.state.retrieval.search_with_trace(SearchRequest(
            user_id="u", query="What is the relationship needle?", options=["choice"], top_k=20))
    budget = trace["optional_routes"]["route_budget"]
    assert budget["planned"]["original"] > 0
    assert budget["planned_total"] <= 3 and budget["actual_total"] <= 3
    assert budget["actual_total"] == trace["optional_routes"]["index_hits"]


def _search_occupancy(tmp_path, text, *, parents=False, atoms=False, display=False, token_budget=8000):
    settings = Settings(db_path=tmp_path / "occupancy.db", profile_id="cycle2_raw",
                        enable_extractor=atoms, atom_endpoint="https://mini.test/chat" if atoms else None,
                        parent_evidence=parents, atom_display=display,
                        return_char_budget=100000, return_token_budget=token_budget)
    app = create_app(settings, atom_provider=LocalAtoms(settings, text="needle") if atoms else None)
    with TestClient(app) as client:
        body = payload(text if isinstance(text, list) else [text])
        assert client.post("/add", json=body).status_code == 200
        result, trace = app.state.retrieval.search_with_trace(
            SearchRequest(user_id="u", query="needle", top_k=20))
    return result, trace


def test_same_source_raw_and_span_use_one_slot_and_budget(tmp_path):
    message = "needle source text."
    for parents in (False, True):
        result, trace = _search_occupancy(tmp_path / str(parents), message, parents=parents, atoms=True,
                                           token_budget=1000)
        assert len(result.data) == 1
        assert result.data[0].content == message
        assert trace["output_characters"] == len(message)
        assert trace["packing"]["packed_content_tokens"] == token_count(message)
        assert trace["evidence"][0]["source_raw_id"] == trace["evidence"][0]["id"] or parents
        assert trace["source_occupancy"]["kept"] >= 1


def test_uncovered_spans_remain_and_covered_spans_collapse(tmp_path):
    filler = "a" * 1700
    message = "needle alpha " + filler + " needle beta " + filler
    needle_spans = [message[start:end] for start, end, _ in boundaries(message) if "needle" in message[start:end]]
    assert len(needle_spans) >= 2 and len({span for span in needle_spans}) >= 2
    result, trace = _search_occupancy(tmp_path / "spans", message, atoms=False, token_budget=16000)
    sources = {item["source_raw_id"] for item in trace["evidence"]}
    assert len(sources) == 1 and len(result.data) >= 2
    assert len({item["id"] for item in trace["evidence"]}) == len(trace["evidence"])
    assert all(item["id"] != item["source_raw_id"] for item in trace["evidence"])
    assert trace["output_characters"] == sum(len(item.content) for item in result.data)
    assert trace["packing"]["packed_content_tokens"] == sum(token_count(item.content) for item in result.data)
    assert trace["source_occupancy"]["dropped_ids"] == []

    covered, covered_trace = _search_occupancy(tmp_path / "raw", message, atoms=True, token_budget=16000)
    assert len(covered.data) == 1
    assert covered.data[0].content == message
    assert covered_trace["evidence"][0]["id"] == covered_trace["evidence"][0]["source_raw_id"]
    assert covered_trace["source_occupancy"]["dropped_ids"]
    assert covered_trace["output_characters"] == len(message)
    assert covered_trace["packing"]["packed_content_tokens"] == token_count(message)
    assert covered_trace["output_characters"] < len(message) + len(result.data[0].content)


def test_distinct_sources_keep_independent_slots(tmp_path):
    result, trace = _search_occupancy(
        tmp_path, ["needle one source.", "needle two source."], atoms=True, token_budget=1000)
    assert len(result.data) == 2
    assert {item["source_raw_id"] for item in trace["evidence"]} == {item.id for item in result.data}
    assert len({item["source_raw_id"] for item in trace["evidence"]}) == 2
    assert trace["output_characters"] == sum(len(item.content) for item in result.data)
    assert trace["packing"]["packed_content_tokens"] == sum(token_count(item.content) for item in result.data)


def test_atom_display_group_still_packs_the_aggregated_raw(tmp_path):
    result, trace = _search_occupancy(tmp_path, "needle source text.", atoms=True, display=True, token_budget=2000)
    assert len(result.data) == 1
    assert "Derived navigation atom" in result.data[0].content
    assert "needle source text." in result.data[0].content
    assert trace["output_characters"] == len(result.data[0].content)
    assert trace["packing"]["packed_content_tokens"] == token_count(result.data[0].content)
    assert trace["atom_display"]["shown_atom_ids"]
