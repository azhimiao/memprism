"""Offline behavior tests; fake scores are not semantic-quality measurements."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import pytest

from app.models import EvidenceCandidate
from app.reranking import EvidenceReranker, MODEL_REVISION, RerankConfig
from app import reranking


def hit(identity, text="source", **fields):
    values = dict(id=identity, user_id="u", content=text, score=0.025,
                  created_at=17, deletion_state="active", source_raw_id=identity, message_index=3)
    return EvidenceCandidate(**(values | fields))


def pair_count(query, document):
    return len(query) + len(document) + 4


def config(**fields):
    return replace(RerankConfig(enabled=True, max_length=32, max_query_tokens=8,
                                window_chars=24, overlap_chars=4, timeout_seconds=1), **fields)


def plugin(scorer, **fields):
    return EvidenceReranker(config(**fields), scorer=scorer, pair_token_counter=pair_count,
                            scorer_id="fixture-v1", tokenizer_id="codepoints-plus-4-v1")


def test_disabled_never_dispatches_or_loads_optional_dependency(monkeypatch):
    def fail(*_):
        pytest.fail("disabled reranking dispatched")
    monkeypatch.setattr(reranking._LocalBackend, "_start", fail)
    candidates = [hit("first"), hit("second")]
    with EvidenceReranker() as ranker:
        ordered, trace = ranker.rerank("query", candidates, user_id="u")
        assert ranker.warmup()["status"] == "disabled"
    assert ordered == candidates and all(a is b for a, b in zip(ordered, candidates))
    assert trace["status"] == "disabled" and trace["batches_started"] == 0
    with EvidenceReranker(scorer=fail) as ranker:
        assert ranker.rerank("query", candidates, user_id="u")[1]["status"] == "disabled"


def test_tail_exception_beats_similar_counterevidence_and_preserves_metadata():
    calls = []
    def scorer(query, texts):
        calls.append((query, list(texts)))
        return [9.0 if "exception=deny" in text else 1.0 for text in texts]
    # This is a scorer contract fixture, not evidence that a real CE understands negation.
    wrong = hit("similar", "exception=allow")
    text = "前提🙂e\u0301" * 20 + "\nexception=deny"
    correct = hit("tail", text, source_raw_id="raw", representation="span",
                  span_start=73, span_end=73 + len(text), source_version=4,
                  source_sha256="raw-hash", span_sha256="span-hash", fragment_hash="policy-hash",
                  role="user", session_id="session", source_timestamp=11, recorded_at=12,
                  boundary_status="source_end", previous_id="before", next_id="after")
    candidates = [wrong, correct]
    before = [candidate.model_dump() for candidate in candidates]
    with plugin(scorer, batch_size=3) as ranker:
        ordered, trace = ranker.rerank("q", candidates, user_id="u")
    assert ordered == [correct, wrong] and ordered[0] is correct
    assert [candidate.model_dump() for candidate in candidates] == before
    assert trace["status"] == "ok" and trace["coverage_complete"]
    assert all(len(texts) <= 3 for _, texts in calls)
    scored_texts = [piece for _, texts in calls for piece in texts]
    assert any("exception=deny" in piece for piece in scored_texts)
    assert len(scored_texts) == trace["windows_scored"]
    expected_texts = []
    for window in trace["windows"]:
        candidate = candidates[window["candidate_index"]]
        expected_texts.append(candidate.content[window["start"]:window["end"]])
        assert window["pair_tokens"] <= 32
        assert window["source_start"] == (candidate.span_start or 0) + window["start"]
        assert window["source_end"] == (candidate.span_start or 0) + window["end"]
    assert scored_texts == expected_texts
    windows = [window for window in trace["windows"] if window["candidate_id"] == "tail"]
    assert windows[0]["start"] == 0 and windows[-1]["end"] == len(text)
    assert all(left["end"] >= right["start"] for left, right in zip(windows, windows[1:]))
    assert trace["candidate_scores"][1]["window_max_score"] == 9
    assert trace["candidate_scores"][1]["retrieval_score"] == 0.025
    assert trace["identity"]["implementation_kind"] == "injected"


def test_parent_max_has_no_vote_bonus_and_support_components_are_atomic():
    values = {"low": -4., "high": 9., "duplicate": 9., "other": 8., "bridge": 1.}
    candidates = [hit("other", "other"), hit("a1", "low", source_raw_id="a", source_version=1),
                  hit("bridge", "bridge"), hit("a2", "high", source_raw_id="a", source_version=1),
                  hit("a3", "duplicate", source_raw_id="a", source_version=1)]
    groups = [{"seed_id": "a1", "member_ids": ["a1", "bridge"]},
              {"seed_id": "bridge", "member_ids": ["bridge", "a2"]}]
    with plugin(lambda _, texts: [values[text] for text in texts]) as ranker:
        ordered, trace = ranker.rerank("q", candidates, user_id="u", groups=groups)
    assert [candidate.id for candidate in ordered] == ["a1", "bridge", "a2", "a3", "other"]
    assert trace["components"][0] == {"member_ids": ["a1", "bridge", "a2", "a3"], "score": 9.}
    assert len(ordered) == len(candidates) and {id(hit) for hit in ordered} == {id(hit) for hit in candidates}


def test_distinct_source_versions_not_merged_and_ties_stable():
    candidates = [hit("b"), hit("a1", source_raw_id="raw", source_version=1),
                  hit("a2", source_raw_id="raw", source_version=2)]
    with plugin(lambda _, texts: [-3.] * len(texts)) as ranker:
        ordered, trace = ranker.rerank("q", candidates, user_id="u")
    assert ordered == candidates
    assert len(trace["components"]) == 3


def test_nonmonotone_counter_windows_are_recounted_and_cover_every_character():
    def nonmonotone(query, document):
        return 5 if not document else (80 if len(document) % 3 == 0 else len(document) + 5)
    candidates = [hit("unicode", "🙂\u0301中\n" * 31)]
    seen = []
    def scorer(query, texts):
        seen.extend(texts)
        assert all(nonmonotone(query, text) <= 32 for text in texts)
        return [0.] * len(texts)
    with EvidenceReranker(config(), scorer=scorer, pair_token_counter=nonmonotone,
                          tokenizer_id="nonmonotone-fixture") as ranker:
        _, trace = ranker.rerank("q", candidates, user_id="u")
    assert trace["status"] == "ok"
    covered = set()
    for window, text in zip(trace["windows"], seen):
        assert text == candidates[0].content[window["start"]:window["end"]]
        covered.update(range(window["start"], window["end"]))
    assert covered == set(range(len(candidates[0].content)))


@pytest.mark.parametrize("fields,texts,reason", [
    ({"max_candidates": 1}, ["a", "b"], "candidate_budget_exceeded"),
    ({"max_candidate_chars": 3}, ["abcd"], "text_budget_exceeded"),
    ({"max_total_chars": 3}, ["ab", "cd"], "text_budget_exceeded"),
    ({"max_windows_per_candidate": 1}, ["x" * 80], "window_budget_exceeded"),
    ({"max_windows": 1}, ["a", "b"], "window_budget_exceeded"),
    ({"max_pair_tokens": 10}, ["a", "b"], "token_budget_exceeded"),
])
def test_preflight_budgets_fallback_without_any_scoring(fields, texts, reason):
    calls = []
    with plugin(lambda *args: calls.append(args), **fields) as ranker:
        candidates = [hit(str(i), text) for i, text in enumerate(texts)]
        ordered, trace = ranker.rerank("q", candidates, user_id="u")
    assert ordered == candidates and calls == []
    assert trace["status"] == "fallback" and trace["reason"] == reason and trace["fallback"]
    assert not trace["coverage_complete"]


@pytest.mark.parametrize("query,reason", [("", "invalid_input"), ("query long", "query_budget_exceeded"),
                                         ("x" * 4100, "query_budget_exceeded")])
def test_query_not_silently_truncated(query, reason):
    calls = []
    with plugin(lambda *args: calls.append(args)) as ranker:
        _, trace = ranker.rerank(query, [hit("a")], user_id="u")
    assert trace["reason"] == reason and calls == []


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [True], [[1.]], [], [1., 2.], ["1"]])
def test_invalid_score_shape_or_value_falls_back(values):
    candidates = [hit("a")]
    with plugin(lambda *_: values) as ranker:
        ordered, trace = ranker.rerank("q", candidates, user_id="u")
    assert ordered == candidates and trace["reason"] == "invalid_scores"
    assert "candidate_scores" not in trace and "windows" not in trace


@pytest.mark.parametrize("counter,reason", [(lambda *_: False, "invalid_token_count"),
                                          (lambda *_: 1.5, "invalid_token_count"),
                                          (lambda _, text: 5 if not text else 100, "window_cannot_fit")])
def test_unusable_tokenizer_cannot_produce_incomplete_rankings(counter, reason):
    with EvidenceReranker(config(), scorer=lambda *_: pytest.fail("must preflight first"),
                          pair_token_counter=counter, tokenizer_id="broken-fixture") as ranker:
        _, trace = ranker.rerank("q", [hit("a")], user_id="u")
    assert trace["reason"] == reason


@pytest.mark.parametrize("error,reason", [(MemoryError("secret text"), "oom"),
                                        (RuntimeError("CUDA out of memory: secret text"), "oom"),
                                        (RuntimeError("secret text"), "scoring_failed"),
                                        (TimeoutError("secret text"), "timeout")])
def test_second_batch_failure_discards_partial_scores_and_private_error_text(error, reason):
    calls = []
    def scorer(query, texts):
        calls.append(texts)
        if len(calls) == 2:
            raise error
        return [100.]
    candidates = [hit("first"), hit("second")]
    with plugin(scorer, batch_size=1) as ranker:
        ordered, trace = ranker.rerank("q", candidates, user_id="u")
    assert ordered == candidates and trace["reason"] == reason
    assert trace["batches_started"] == 2 and trace["windows_scored"] == 1
    assert "candidate_scores" not in trace and "secret text" not in json.dumps(trace)


def test_injected_timeout_returns_promptly_and_does_not_accumulate_jobs():
    entered, release = threading.Event(), threading.Event()
    calls = []
    def scorer(query, texts):
        calls.append(texts)
        entered.set()
        release.wait(2)
        return [1.] * len(texts)
    with plugin(scorer, timeout_seconds=0.1, batch_size=1) as ranker:
        started = time.monotonic()
        candidates = [hit("a"), hit("b")]
        ordered, trace = ranker.rerank("q", candidates, user_id="u")
        try:
            assert entered.is_set() and time.monotonic() - started < 1
            assert ordered == candidates and trace["reason"] == "timeout"
            assert not trace["usage_complete"]
            assert ranker.rerank("q", candidates, user_id="u")[1]["reason"] == "busy"
            assert len(calls) == 1
        finally:
            release.set()
            assert ranker.backend.lock.acquire(timeout=2)
            ranker.backend.lock.release()
        assert len(calls) == 1  # Expired jobs cannot dispatch the next batch.


def test_caller_timeout_stops_next_batch_even_before_clock_deadline(monkeypatch):
    from types import SimpleNamespace

    entered, release = threading.Event(), threading.Event()
    calls = []

    def scorer(query, texts):
        calls.append(texts)
        entered.set()
        assert release.wait(2)
        return [1.] * len(texts)

    # Keep the deadline clock before expiry while Event.wait reaches its timeout.
    # This makes the boundary disagreement reproducible without a timing race.
    monkeypatch.setattr(reranking, "time", SimpleNamespace(monotonic=lambda: 0.0))
    with plugin(scorer, timeout_seconds=0.05, batch_size=1) as ranker:
        candidates = [hit("a"), hit("b")]
        try:
            ordered, trace = ranker.rerank("q", candidates, user_id="u")
            assert entered.is_set()
            assert ordered == candidates and trace["reason"] == "timeout"
            assert ranker.rerank("q", candidates, user_id="u")[1]["reason"] == "busy"
        finally:
            release.set()
            assert ranker.backend.lock.acquire(timeout=2)
            ranker.backend.lock.release()
        assert len(calls) == 1


@pytest.mark.parametrize("candidates,groups,reason", [
    ([hit("a", user_id="other")], (), "invalid_candidates"),
    ([hit("a", deletion_state="deleted")], (), "invalid_candidates"),
    ([hit("a", " ")], (), "invalid_candidates"),
    ([hit("a"), hit("a")], (), "duplicate_candidates"),
    ([hit("a")], ({"seed_id": "a", "member_ids": ["missing"]},), "missing_dependency"),
    ([hit("a")], ({"seed_id": "a", "member_ids": ["a"], "unavailable_dependencies": ["x"]},), "missing_dependency"),
    ([hit("a")], ({"seed_id": "a", "member_ids": "a"},), "invalid_groups"),
])
def test_invalid_pool_or_incomplete_support_never_reaches_scorer(candidates, groups, reason):
    with plugin(lambda *_: pytest.fail("invalid pool dispatched")) as ranker:
        ordered, trace = ranker.rerank("q", candidates, user_id="u", groups=groups)
    assert ordered == candidates and trace["reason"] == reason


def test_empty_pool_and_closed_backend_do_not_dispatch():
    ranker = plugin(lambda *_: pytest.fail("unexpected dispatch"))
    assert ranker.rerank("q", [], user_id="u")[1]["status"] == "empty"
    ranker.close()
    assert ranker.rerank("q", [hit("a")], user_id="u")[1]["reason"] == "closed"


def test_offline_surrogate_identity_is_distinct_and_custom_tokenizer_requires_identity():
    with EvidenceReranker(RerankConfig(enabled=True), scorer=lambda _, texts: [0.] * len(texts)) as ranker:
        _, trace = ranker.rerank("问", [hit("unicode", "尾部🙂" * 100)], user_id="u")
    assert trace["status"] == "ok"
    assert trace["identity"]["tokenizer_id"] == "offline_utf8_bytes_plus_4_v1"
    assert trace["identity"]["model"] is None and trace["identity"]["revision"] is None
    with pytest.raises(ValueError, match="tokenizer_id"):
        EvidenceReranker(scorer=lambda *_: [], pair_token_counter=pair_count)


@pytest.mark.parametrize("fail", [False, True])
def test_create_app_injected_scorer_runs_after_gate_and_records_http_trace(tmp_path, caplog, fail):
    import logging
    import sqlite3
    from fastapi.testclient import TestClient
    from app.api import create_app
    from app.config import Settings
    from app.models import SearchRequest

    calls = []
    def scorer(query, documents):
        calls.extend(documents)
        assert all("secret-forbidden" not in text and "outsider-private" not in text for text in documents)
        if fail:
            raise MemoryError("not logged")
        return [10. if "exception=deny" in text else 0. for text in documents]

    tail = "policy " + "details " * 140 + "exception=deny"
    similar = "policy " * 20 + "allow"
    path = tmp_path / "service.sqlite"
    with EvidenceReranker(RerankConfig(enabled=True), scorer=scorer, scorer_id="http-fixture") as ranker:
        app = create_app(Settings(db_path=path), reranker=ranker)
        with TestClient(app) as client:
            for user, texts in (("u", [similar, tail, "policy secret-forbidden"]),
                                ("other", ["policy outsider-private"])):
                assert client.post("/add", json={"request_id": user, "user_id": user, "session_id": "s",
                                                  "messages": [{"role": "user", "content": text} for text in texts]}).status_code == 200
            with sqlite3.connect(path) as db:
                db.execute("UPDATE source_policies SET disclosure_policy='blocked' WHERE raw_id IN "
                           "(SELECT id FROM raw_messages WHERE content=?)", ("policy secret-forbidden",))
            request = SearchRequest(query="policy", user_id="u", top_k=2)
            app.state.retrieval.reranker = None
            baseline, baseline_trace = app.state.retrieval.search_with_trace(request)
            app.state.retrieval.reranker = ranker
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="aml.retrieval"):
                response = client.post("/search", json=request.model_dump())
            assert response.status_code == 200
            records = [record.getMessage() for record in caplog.records if record.name == "aml.retrieval"]
            trace = json.loads(records[-1].removeprefix("retrieval_trace "))
            assert {item["id"] for item in response.json()["data"]} == {hit.id for hit in baseline.data}
            assert {item["candidate_id"] for item in trace["reranking"].get("candidate_scores", [])} <= set(baseline_trace["returned_ids"])
            if fail:
                assert trace["reranking"]["status"] == "fallback" and trace["reranking"]["reason"] == "oom"
                assert response.json() == baseline.model_dump()
            else:
                assert trace["reranking"]["status"] == "ok" and response.json()["data"][0]["content"] == tail
                baseline_scores = {hit.id: hit.score for hit in baseline.data}
                assert all(item["score"] == baseline_scores[item["id"]] for item in response.json()["data"])
                assert trace["reranking"]["windows_scored"] > 2
            assert calls and any(item["reason"] == "disclosure_blocked" for item in trace["gate_decisions"])


def _ready_but_stuck_worker(connection, config):
    connection.send({"status": "ready", "runtime": {"warmup_complete": True}})
    connection.recv()
    threading.Event().wait(30)


def _startup_stuck_worker(connection, config):
    threading.Event().wait(30)


def _crashing_worker(connection, config):
    connection.send({"status": "ready", "runtime": {"warmup_complete": True}})
    connection.recv()
    connection.close()


def _oom_worker(connection, config):
    connection.send({"status": "ready", "runtime": {"warmup_complete": True}})
    connection.recv()
    connection.send({"status": "error", "reason": "oom", "usage": {"batches_started": 1}})
    connection.close()


def _ready_without_reading_worker(connection, config):
    connection.send({"status": "ready", "runtime": {"warmup_complete": True}})
    threading.Event().wait(30)


def _successful_worker(connection, config):
    connection.send({"status": "ready", "runtime": {"warmup_complete": True}})
    try:
        while True:
            request = connection.recv()
            result = reranking._evaluate(request["query"], request["documents"], config,
                                          lambda _, texts: [2. if "tail" in text else -1. for text in texts],
                                          pair_count, time.monotonic() + config.timeout_seconds, reranking._usage())
            connection.send({"status": "ok", "result": result})
    except EOFError:
        connection.close()


def _local_test_config(tmp_path, **fields):
    path = tmp_path / MODEL_REVISION
    path.mkdir(exist_ok=True)
    return replace(RerankConfig(enabled=True, local_model_path=path, startup_timeout_seconds=5,
                                timeout_seconds=0.15, shutdown_timeout_seconds=0.3), **fields)


@pytest.mark.parametrize("target,reason", [(_ready_but_stuck_worker, "timeout"),
                                          (_crashing_worker, "worker_failed"), (_oom_worker, "oom")])
def test_spawned_local_worker_failure_is_killed_and_order_preserved(tmp_path, monkeypatch, target, reason):
    monkeypatch.setattr(reranking, "_local_worker", target)
    with EvidenceReranker(_local_test_config(tmp_path)) as ranker:
        assert ranker.warmup()["status"] == "ready"
        process = ranker.backend.process
        pid = process.pid
        candidates = [hit("a"), hit("b")]
        ordered, trace = ranker.rerank("q", candidates, user_id="u")
        assert ordered == candidates and trace["reason"] == reason
        assert ranker.backend.process is None and ranker.backend.connection is None
        assert process._closed  # join and close completed; no orphan worker handle.
        assert pid is not None and trace["identity"]["implementation_kind"] == "local_cross_encoder"
        assert ranker.warmup()["status"] == "ready"  # New request, not a retry of failed inference.
        assert ranker.backend.generation == 2


def test_local_startup_timeout_is_bounded_and_does_not_leave_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(reranking, "_local_worker", _startup_stuck_worker)
    with EvidenceReranker(_local_test_config(tmp_path, startup_timeout_seconds=0.15)) as ranker:
        started = time.monotonic()
        trace = ranker.warmup()
        assert time.monotonic() - started < 2
        assert trace["status"] == "failed" and trace["reason"] == "startup_timeout"
        assert ranker.backend.process is None


def test_local_pipe_write_is_also_inside_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(reranking, "_local_worker", _ready_without_reading_worker)
    with EvidenceReranker(_local_test_config(tmp_path)) as ranker:
        assert ranker.warmup()["status"] == "ready"
        # Larger than a pipe buffer; send() must not hang outside the timeout.
        candidates = [hit(str(i), "x" * 60_000) for i in range(4)]
        started = time.monotonic()
        ordered, trace = ranker.rerank("q", candidates, user_id="u")
        assert time.monotonic() - started < 2
        assert ordered == candidates and trace["reason"] == "timeout"
        assert not ranker.backend.sender.is_alive() and ranker.backend.process is None


def test_local_worker_is_reused_and_successful_requests_have_separate_traces(tmp_path, monkeypatch):
    monkeypatch.setattr(reranking, "_local_worker", _successful_worker)
    with EvidenceReranker(_local_test_config(tmp_path, timeout_seconds=1)) as ranker:
        assert ranker.warmup()["status"] == "ready"
        pid = ranker.backend.process.pid
        candidates = [hit("base"), hit("tail", "x" * 1500 + "tail")]
        ordered, first = ranker.rerank("q", candidates, user_id="u")
        assert [candidate.id for candidate in ordered] == ["tail", "base"] and first["status"] == "ok"
        _, second = ranker.rerank("q", [hit("other")], user_id="u")
        assert ranker.backend.process.pid == pid and ranker.backend.generation == 1
        assert second["status"] == "ok" and second["windows_scored"] == 1
        assert len(first["windows"]) > len(second["windows"])


def test_process_start_failure_is_cleaned_up_as_fallback(tmp_path, monkeypatch):
    class BrokenProcess:
        pid = None
        def start(self): raise OSError("private startup details")
        def close(self): self.closed = True
    context = reranking.mp.get_context("spawn")
    process = BrokenProcess()
    monkeypatch.setattr(reranking.mp, "get_context", lambda _: type("Context", (), {
        "Pipe": staticmethod(context.Pipe), "Process": staticmethod(lambda **_: process)})())
    with EvidenceReranker(_local_test_config(tmp_path)) as ranker:
        ordered, trace = ranker.rerank("q", [hit("a")], user_id="u")
        assert ordered[0].id == "a" and trace["reason"] == "worker_failed"
        assert process.closed and ranker.backend.process is None
        assert "private startup details" not in json.dumps(trace)


@pytest.mark.parametrize("path,reason", [(None, "missing_configuration"),
                                        ("wrong-revision", "model_revision_mismatch"),
                                        (MODEL_REVISION, "model_missing")])
def test_local_missing_snapshot_fails_without_downloading_or_spawning(tmp_path, path, reason):
    model_path = None if path is None else tmp_path / path
    with EvidenceReranker(RerankConfig(enabled=True, local_model_path=model_path)) as ranker:
        trace = ranker.warmup()
        assert trace["reason"] == reason
        assert ranker.backend.process is None and ranker.backend.generation == 0


def test_local_loader_pins_device_tokenizer_weights_versions_and_prediction(tmp_path, monkeypatch):
    """API contract stub only. No torch/ST installation or model download."""
    import sys
    import types
    import hashlib
    path = tmp_path / MODEL_REVISION
    path.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors", "special_tokens_map.json"):
        (path / name).write_bytes(b"fixture")
    monkeypatch.setattr(reranking, "PINNED_ARTIFACTS", {
        name: ("sha256", hashlib.sha256(b"fixture").hexdigest()) for name in reranking.PINNED_ARTIFACTS})
    calls = {}
    class Identity:
        pass
    class Tokenizer:
        def __call__(self, query, document, **kwargs):
            calls["tokenizer"] = (query, document, kwargs)
            return {"input_ids": list(range(len(query) + len(document) + 4))}
    class CrossEncoder:
        def __init__(self, path, **kwargs):
            calls["load"] = (path, kwargs)
            self.tokenizer = Tokenizer()
            self.model = types.SimpleNamespace(eval=lambda: calls.update(eval=True))
        def predict(self, pairs, **kwargs):
            calls["predict"] = (pairs, kwargs)
            return [0.5] * len(pairs)
    class InferenceMode:
        def __enter__(self): pass
        def __exit__(self, *_): pass
    torch = types.SimpleNamespace(float32="fp32", nn=types.SimpleNamespace(Identity=Identity),
                                  set_num_threads=lambda n: calls.update(threads=n),
                                  set_num_interop_threads=lambda n: None, manual_seed=lambda n: None,
                                  use_deterministic_algorithms=lambda n: None, inference_mode=InferenceMode)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=CrossEncoder))
    monkeypatch.setattr(reranking, "version", lambda name: reranking.LOCAL_VERSIONS[name])
    # Restore worker-only environment changes after this in-process contract stub.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM",
                 "CUBLAS_WORKSPACE_CONFIG"):
        monkeypatch.setenv(name, "fixture-before")
    scorer, counter, identity = reranking._load_cross_encoder(RerankConfig(enabled=True, local_model_path=path))
    assert calls["load"][0] == str(path)
    settings = calls["load"][1]
    assert settings["local_files_only"] is True and settings["trust_remote_code"] is False
    assert settings["revision"] == MODEL_REVISION and settings["token"] is False
    assert settings["device"] == "cpu" and settings["max_length"] == 512
    assert settings["model_kwargs"]["use_safetensors"] is True
    assert settings["model_kwargs"]["torch_dtype"] == "fp32"
    assert settings["tokenizer_kwargs"] == {"model_max_length": 512, "use_fast": True}
    assert counter(" q ", " 尾部 ") == 11
    assert calls["tokenizer"] == (" q ", " 尾部 ", {"add_special_tokens": True, "truncation": False})
    assert scorer("q", ["a", "b"]) == [0.5, 0.5]
    assert calls["predict"][1]["batch_size"] == 8
    assert calls["predict"][1]["apply_softmax"] is False
    assert identity["warmup_complete"] and len(identity["artifact_sha256"]) == 5


def test_renaming_arbitrary_model_to_pinned_revision_cannot_impersonate_model(tmp_path):
    path = tmp_path / MODEL_REVISION
    path.mkdir()
    for name in reranking.PINNED_ARTIFACTS:
        (path / name).write_bytes(b"wrong-model-fixture")
    with pytest.raises(reranking.RerankFailure, match="model_artifact_mismatch"):
        reranking._artifact_identity(path)


@pytest.mark.parametrize("fields", [{"max_length": 513}, {"max_query_tokens": 512}, {"device": "auto"},
                                    {"device": "cuda"}, {"batch_size": 0}, {"max_candidates": 201},
                                    {"overlap_chars": 1200}, {"timeout_seconds": float("nan")},
                                    {"timeout_seconds": float("inf")}, {"window_chars": True}])
def test_invalid_runtime_limits_are_rejected_before_dispatch(fields):
    with pytest.raises(ValueError):
        RerankConfig(**fields)


def test_trace_is_a_snapshot_not_mutable_state_shared_with_later_requests():
    with plugin(lambda _, texts: [0.] * len(texts)) as ranker:
        _, first = ranker.rerank("q", [hit("a")], user_id="u")
        ranker.backend.runtime = {"artifact_sha256": {"model": "original-hash"}}
        warmup = ranker.warmup()
        first["identity"]["max_length"] = 999
        warmup["runtime"]["artifact_sha256"]["model"] = "edited-hash"
        _, second = ranker.rerank("q", [hit("b")], user_id="u")
        assert second["identity"]["max_length"] == 32
        assert second["runtime"]["artifact_sha256"]["model"] == "original-hash"
