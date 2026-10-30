"""Synthetic long-query regressions: real HTTP adapter/SQLite, no external calls."""
import json
import math
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.embedding_accounting import AccountedEmbeddingProvider
from app.embeddings import (EmbeddingConfig, EmbeddingFailure, HttpEmbeddingProvider,
                            embed_retrieval_query, provider_version)
from app.jobs import NotReady
from app.models import SearchRequest


def native(monkeypatch, *, limit=8192, batch_size=10, handler=None):
    monkeypatch.setenv('AML_EMBEDDING_API_KEY', 'synthetic-key')
    cfg = EmbeddingConfig(provider='dashscope',
        endpoint='https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding',
        dimensions=64, max_input_bytes=limit, batch_size=batch_size)
    wire = []
    def respond(request):
        body = json.loads(request.content)
        wire.append(body)
        if handler is not None:
            return handler(body, len(wire))
        return httpx.Response(200, json={'output': {'embeddings': [
            {'text_index': i, 'embedding': [1., float(i)] + [0.] * 62}
            for i in range(len(body['input']['texts']))]}, 'usage': {'total_tokens': 3}})
    return HttpEmbeddingProvider(cfg, transport=httpx.MockTransport(respond)), wire


def inputs(wire):
    return [text for body in wire for text in body['input']['texts']]


def test_observed_shape_old_guard_rejects_new_query_path_covers_all_text(monkeypatch):
    provider, wire = native(monkeypatch)
    text = 'a' * 9004 + 'é' * 46
    assert len(text) == 9050 and len(text.encode()) == 9096
    with pytest.raises(EmbeddingFailure, match='input_too_long'):
        provider.for_query().embed([text])
    assert wire == []
    identity = provider_version(provider)
    batch = embed_retrieval_query(provider.for_query(), text)
    assert ''.join(inputs(wire)) == text
    assert [len(t.encode()) for t in inputs(wire)] == [8192, 904]
    assert len(wire) == 1 and wire[0]['parameters']['text_type'] == 'query'
    assert len(batch.vectors) == 1 and sum(v*v for v in batch.vectors[0]) == pytest.approx(1)
    assert batch.usage['query_window_count'] == 2 and batch.usage['requests'] == 1
    assert provider_version(provider) == identity
    # The document path and its persisted model identity remain strict.
    with pytest.raises(EmbeddingFailure, match='input_too_long'):
        provider.embed([text])
    assert len(wire) == 1


@pytest.mark.parametrize('text', ['x' * 8192, '中' * 2730, '🙂' * 2048], ids=['ascii-limit', 'cjk-short', 'emoji-limit'])
def test_short_and_boundary_query_is_exact_single_input(monkeypatch, text):
    provider, wire = native(monkeypatch)
    batch = embed_retrieval_query(provider.for_query(), text)
    assert inputs(wire) == [text] and len(batch.vectors) == 1
    assert 'query_input_policy' not in batch.usage


@pytest.mark.parametrize('text', ['中🙂é' * 2000, 'x' * 8191 + '🙂tail',
                                  'x' * 8192 + ' ' * 17000 + '尾部'], ids=['mixed', 'split-emoji', 'blank-window'])
def test_utf8_boundaries_and_whitespace_windows_keep_all_semantic_text(monkeypatch, text):
    provider, wire = native(monkeypatch)
    embed_retrieval_query(provider.for_query(), text)
    sent = inputs(wire)
    assert all(0 < len(t.encode()) <= 8192 and t.strip() for t in sent)
    assert ''.join(''.join(sent).split()) == ''.join(text.split())


def test_unicode_preflight_prevents_any_dispatch(monkeypatch):
    provider, wire = native(monkeypatch)
    with pytest.raises(EmbeddingFailure, match='invalid_input_unicode'):
        embed_retrieval_query(provider.for_query(), 'x' * 9000 + '\ud800')
    assert wire == []


def test_total_window_budget_is_checked_before_first_dispatch(monkeypatch):
    provider, wire = native(monkeypatch, limit=16)
    with pytest.raises(EmbeddingFailure) as error:
        embed_retrieval_query(provider.for_query(), 'x' * 161)
    assert error.value.code == 'input_too_long'
    assert error.value.usage['query_window_budget_exceeded'] is True
    assert error.value.usage['requests'] == 0 and wire == []


def test_existing_rate_gate_can_fail_multibatch_query_without_hiding_first_cost(monkeypatch, tmp_path):
    from dataclasses import replace
    provider, wire = native(monkeypatch, limit=16, batch_size=1)
    provider.config = replace(provider.config, min_request_interval_seconds=60)
    accounted = AccountedEmbeddingProvider(provider, tmp_path/'rate.sqlite').for_query()
    with pytest.raises(EmbeddingFailure) as error:
        embed_retrieval_query(accounted, 'x' * 33)
    assert error.value.code == 'embedding_rate_limited'
    assert len(wire) == 1 and error.value.usage['requests'] == 1
    assert error.value.usage['reported_input_tokens'] == 3


def test_every_chunk_dispatch_remains_authorized_and_partial_cost_survives(monkeypatch, tmp_path):
    provider, wire = native(monkeypatch, limit=16, batch_size=1)
    accounted = AccountedEmbeddingProvider(provider, tmp_path/'ledger.sqlite').for_query()
    checks = []
    def authorize():
        checks.append(True)
        if len(checks) == 2:
            raise NotReady('query_config_generation_changed')
    with pytest.raises(NotReady):
        embed_retrieval_query(accounted, 'abcde' * 10, before_dispatch=authorize)
    assert len(wire) == 1 and len(checks) == 2
    with sqlite3.connect(accounted.ledger_path) as db:
        assert db.execute("SELECT COUNT(*) FROM embedding_calls WHERE status='completed'").fetchone()[0] == 1


def test_partial_embedding_failure_keeps_usage_and_never_fakes_success(monkeypatch, tmp_path):
    def respond(body, number):
        if number == 2:
            return httpx.Response(503, json={'message': 'private failure'})
        return httpx.Response(200, json={'output': {'embeddings': [
            {'text_index': 0, 'embedding': [1.] + [0.] * 63}]}, 'usage': {'total_tokens': 7}})
    provider, wire = native(monkeypatch, limit=16, batch_size=1, handler=respond)
    accounted = AccountedEmbeddingProvider(provider, tmp_path/'ledger.sqlite').for_query()
    with pytest.raises(EmbeddingFailure) as error:
        embed_retrieval_query(accounted, 'x' * 40)
    assert len(wire) == 2 and error.value.usage['requests'] == 2
    assert error.value.usage['reported_input_tokens'] == 7


def test_opposing_windows_return_finite_deterministic_direction(monkeypatch):
    def respond(body, number):
        return httpx.Response(200, json={'output': {'embeddings': [
            {'text_index': i, 'embedding': [(-1.)**i] + [0.] * 63}
            for i in range(len(body['input']['texts']))]}, 'usage': {'total_tokens': 1}})
    provider, _ = native(monkeypatch, limit=16, handler=respond)
    result = embed_retrieval_query(provider.for_query(), 'a' * 32)
    assert result.usage['query_mean_cancelled'] is True
    assert result.vectors[0] == [1.] + [0.] * 63


def test_formal_search_long_query_200_raw_fts_unchanged_and_other_user_isolated(monkeypatch, tmp_path, capsys):
    provider, wire = native(monkeypatch)
    settings = Settings(db_path=tmp_path/'memory.sqlite', profile_id='cycle2_dense',
        embedding=provider.config, retrieval_mode='raw_hybrid_rrf')
    app = create_app(settings, embedding_provider=provider)
    text = 'context ' * 1200 + 'needle_tail'
    with TestClient(app) as client:
        added = client.post('/add', json={'user_id': 'u', 'session_id': 's', 'request_id': 'r',
            'messages': [{'role': 'user', 'content': 'needle_tail original evidence'}]})
        assert added.status_code == 200
        documents = inputs(wire)
        assert documents == ['needle_tail original evidence']
        wire.clear()
        # Capture original full query at the lexical boundary as well as actual wire.
        import app.retrieval as retrieval
        original = retrieval.build_match_query
        seen = []
        def lexical(query):
            seen.append(query)
            return original(query)
        monkeypatch.setattr(retrieval, 'build_match_query', lexical)
        result = client.post('/search', json={'user_id': 'u', 'query': text, 'top_k': 100})
        assert result.status_code == 200 and result.json()['data'][0]['content'] == documents[0]
        assert ''.join(inputs(wire)) == text and text in seen
        assert client.post('/search', json={'user_id': 'other', 'query': text, 'top_k': 100}).json() == {'data': []}
        assert 'query_input_policy' in app.state.retrieval.search_with_trace(SearchRequest(user_id='u', query=text, top_k=100))[1]['query_embedding_usage']
    output = capsys.readouterr()
    assert 'needle_tail' not in output.err and 'synthetic-key' not in output.err
