"""Real ASGI/SQLite with synthetic failure injection; never calls a supplier."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app import diagnostics as d
from app.api import create_app
from app.config import Settings
from app.embeddings import EmbeddingConfig, EmbeddingFailure, HttpEmbeddingProvider
from app.jobs import NotReady
from app.retrieval import LifecycleSnapshotBusy
from app.transport import TransportGuard

PRIVATE = 'SYNTHETIC_PRIVATE_CONTENT_057'


def lines(capsys):
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert PRIVATE not in text
    return [json.loads(line[9:]) for line in text.splitlines() if line.startswith('AML_DIAG ')]


def client(tmp_path, **kw):
    return TestClient(create_app(Settings(db_path=tmp_path/'m.sqlite', profile_id='cycle2_raw', **kw)))


def test_http_query_and_user_text_never_enter_logs_and_request_ids_correlate(tmp_path, capsys):
    c = client(tmp_path)
    payload = {'user_id': PRIVATE, 'query': PRIVATE, 'top_k': 100, 'options': [PRIVATE]}
    response = c.post('/search', json=payload, headers={'x-request-id': PRIVATE})
    assert response.status_code == 200 and response.json() == {'data': []}
    request_id = response.headers['x-aml-diagnostic-id']
    assert len(request_id) == 32 and request_id != PRIVATE
    rows = lines(capsys)
    assert {r['request_id'] for r in rows} == {request_id}
    assert rows[0]['event'] == 'request_started' and rows[-1]['event'] == 'request_finished'
    assert rows[-1]['status'] == 200 and rows[-1]['response_complete']
    assert any(r.get('query_chars') == len(PRIVATE) for r in rows)
    assert d.current_id() is None


@pytest.mark.parametrize('error,code,kind', [
    (NotReady('embedding_http_429'), 'embedding_http_429', 'NotReady'),
    (sqlite3.OperationalError(PRIVATE), 'control_not_ready', 'OperationalError'),
    (ValueError(PRIVATE), 'search_failed', 'ValueError'),
    (LifecycleSnapshotBusy(PRIVATE), 'lifecycle snapshot busy', 'LifecycleSnapshotBusy'),
])
def test_original_503_is_unchanged_but_type_stage_and_source_survive(tmp_path, capsys, error, code, kind):
    c = client(tmp_path)
    def fail(request):
        d.stage('search_snapshot')
        raise error
    c.app.state.retrieval.search_with_trace = fail
    response = c.post('/search', json={'user_id': PRIVATE, 'query': PRIVATE, 'top_k': 100})
    assert response.status_code == 503 and response.json() == {'detail': code}
    assert response.headers['retry-after'] == '1'
    rows = lines(capsys)
    failure = next(r for r in rows if r['event'] == 'operation_failed')
    assert failure['reason'] == code and failure['exception_type'] == kind
    assert failure['stage'] == 'search_snapshot'
    assert any(f['module'] == 'app.api' and f['line'] > 0 for f in failure['frames'])
    assert rows[-1]['status'] == 503 and rows[-1]['retry_after_seconds'] == 1


def test_notready_with_arbitrary_message_is_redacted_in_logs(tmp_path, capsys):
    c = client(tmp_path)
    def fail(request): raise NotReady(PRIVATE)
    c.app.state.retrieval.search_with_trace = fail
    assert c.post('/search', json={'user_id': PRIVATE, 'query': PRIVATE, 'top_k': 100}).status_code == 503
    assert any(r.get('reason') == 'unrecognized_reason' for r in lines(capsys))


def test_auth_and_schema_failures_get_status_without_headers_or_body(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv('AML_SERVICE_API_KEY', PRIVATE)
    c = client(tmp_path, auth_mode='bearer')
    assert c.post('/search', json={'bad': PRIVATE}).status_code == 401
    assert c.post('/search', json={'bad': PRIVATE}, headers={'authorization': 'Bearer ' + PRIVATE}).status_code == 422
    rows = lines(capsys)
    assert [r['status'] for r in rows if r['event'] == 'request_finished'] == [401, 422]
    assert any(r.get('reason') == 'unauthorized' for r in rows)


def test_add_commit_replay_and_visibility_preserved_with_logging(tmp_path, capsys):
    c = client(tmp_path)
    add = {'request_id': PRIVATE, 'user_id': PRIVATE, 'session_id': PRIVATE,
           'messages': [{'role': 'user', 'content': PRIVATE}]}
    assert c.post('/add', json=add).json() == c.post('/add', json=add).json()
    result = c.post('/search', json={'user_id': PRIVATE, 'query': PRIVATE, 'top_k': 100})
    assert result.status_code == 200 and result.json()['data'][0]['content'] == PRIVATE
    assert c.post('/search', json={'user_id': 'different', 'query': PRIVATE, 'top_k': 100}).json() == {'data': []}
    rows = lines(capsys)
    assert len({r['request_id'] for r in rows}) == 4


def test_embedding_failure_before_conversion_and_provider_status_logged(monkeypatch, capsys):
    monkeypatch.setenv('DIAG_TEST_KEY', PRIVATE)
    config = EmbeddingConfig(provider='openai-compatible', endpoint='https://example.test/embeddings', dimensions=8,
                             api_key_env='DIAG_TEST_KEY', max_retries=0)
    provider = HttpEmbeddingProvider(config, transport=httpx.MockTransport(
        lambda request: httpx.Response(429, text=PRIVATE)))
    token = d.begin('/search')
    try:
        with pytest.raises(EmbeddingFailure) as caught:
            provider.embed([PRIVATE])
        assert caught.value.code == 'embedding_http_429'
    finally:
        d.end(token)
    rows = lines(capsys)
    assert any(r['event'] == 'embedding_http' and r['status'] == 429 for r in rows)
    assert any(r['event'] == 'stage_failed' and r['reason'] == 'embedding_http_429'
               and r['exception_type'] == 'EmbeddingFailure' for r in rows)


def test_timeout_exception_type_is_kept_without_exception_text(monkeypatch, capsys):
    monkeypatch.setenv('DIAG_TEST_KEY', PRIVATE)
    config = EmbeddingConfig(provider='openai-compatible', endpoint='https://example.test/embeddings', dimensions=8,
                             api_key_env='DIAG_TEST_KEY', max_retries=0)
    def fail(request): raise httpx.ReadTimeout(PRIVATE, request=request)
    provider = HttpEmbeddingProvider(config, transport=httpx.MockTransport(fail))
    token = d.begin('/search')
    try:
        with pytest.raises(EmbeddingFailure): provider.embed([PRIVATE])
    finally:
        d.end(token)
    rows = lines(capsys)
    assert any(r['event'] == 'embedding_transport_failed' and r['exception_type'] == 'ReadTimeout' for r in rows)
    assert any(r.get('reason') == 'embedding_timeout' for r in rows)


def test_logging_sink_failure_does_not_break_add_or_search(tmp_path, monkeypatch):
    c = client(tmp_path)
    class BrokenSink:
        def write(self, text): raise OSError(PRIVATE)
        def flush(self): raise OSError(PRIVATE)
    monkeypatch.setattr(d.sys, 'stderr', BrokenSink())
    add = {'request_id': 'r', 'user_id': 'u', 'session_id': 's', 'messages': [{'role': 'user', 'content': PRIVATE}]}
    assert c.post('/add', json=add).status_code == 200
    assert c.post('/search', json={'user_id': 'u', 'query': PRIVATE, 'top_k': 100}).status_code == 200


def test_capacity_503_is_observed_outside_fastapi(tmp_path, capsys):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        async def slow(scope, receive, send):
            entered.set()
            await release.wait()
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'{}'})
        gate = TransportGuard(slow, Settings(db_path=tmp_path/'none', max_concurrent_requests=1))
        async def receive(): return {'type': 'http.request', 'body': b'{}'}
        first, second = [], []
        async def send_first(event): first.append(event)
        async def send_second(event): second.append(event)
        scope = {'type': 'http', 'path': '/search', 'headers': []}
        task = asyncio.create_task(gate(scope, receive, send_first))
        await entered.wait()
        await gate(scope, receive, send_second)
        release.set(); await task
        assert second[0]['status'] == 503 and gate.active == 0
        assert dict(second[0]['headers'])[b'retry-after'] == b'1'
    asyncio.run(scenario())
    rows = lines(capsys)
    ended = [r for r in rows if r['event'] == 'request_finished']
    assert {r['status'] for r in ended} == {200, 503}
    assert len({r['request_id'] for r in ended}) == 2
    assert next(r for r in ended if r['status'] == 503)['reason'] == 'service capacity reached'
    rejection = next(r for r in rows if r.get('rejection_type') == 'capacity')
    assert rejection['exception_type'] == 'none' and rejection['branch_module'] == 'app.transport'
    assert rejection['branch_line'] > 0


def test_context_isolation_across_threads_and_allowlist(capsys):
    def work(_):
        token = d.begin('/search')
        try:
            d.stage(PRIVATE)
            d.emit('input_shape', query=PRIVATE, user_id=PRIVATE, reason=PRIVATE, body_bytes=7)
            return d.current_id()
        finally: d.end(token)
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(work, range(16)))
    assert len(set(ids)) == 16
    rows = lines(capsys)
    assert len(rows) == 16 and all(r['stage'] == 'transport' for r in rows)
    assert all(r['reason'] == 'unrecognized_reason' and r['body_bytes'] == 7 for r in rows)


@pytest.mark.parametrize('bad_json', [True, False])
def test_original_embedding_parse_and_dimension_failure_frames_survive(monkeypatch, capsys, bad_json):
    monkeypatch.setenv('DIAG_TEST_KEY', PRIVATE)
    config = EmbeddingConfig(provider='unit-test', endpoint='https://example.test/embeddings',
        dimensions=8, api_key_env='DIAG_TEST_KEY', max_retries=0)
    def respond(request):
        if bad_json: return httpx.Response(200, text=PRIVATE)
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1]}]})
    provider = HttpEmbeddingProvider(config, transport=httpx.MockTransport(respond))
    token = d.begin('/search')
    try:
        with pytest.raises(EmbeddingFailure): provider.embed([PRIVATE])
    finally:
        d.end(token)
    rows = lines(capsys)
    original = next(r for r in rows if r['event'] == 'embedding_transport_failed')
    assert original['exception_type'] == ('JSONDecodeError' if bad_json else 'EmbeddingFailure')
    assert original['frames']
    if not bad_json:
        assert len([f for f in original['frames'] if f['module'] == 'app.embeddings']) >= 2


def test_exception_metadata_property_cannot_replace_original_failure(capsys):
    class UnusualError(Exception):
        @property
        def code(self): raise TypeError(PRIVATE)
        @property
        def sqlite_errorcode(self): raise TypeError(PRIVATE)
    original = UnusualError(PRIVATE)
    @d.observe('embedding')
    def fail(): raise original
    token = d.begin('/search')
    try:
        with pytest.raises(UnusualError) as caught: fail()
        assert caught.value is original
    finally:
        d.end(token)
    assert any(r['event'] == 'stage_failed' for r in lines(capsys))


@pytest.mark.parametrize('kind', ['overflow', 'unicode'])
def test_vector_overflow_and_invalid_unicode_original_types_kept(monkeypatch, capsys, kind):
    monkeypatch.setenv('DIAG_TEST_KEY', PRIVATE)
    config = EmbeddingConfig(provider='dashscope', endpoint='https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding',
        dimensions=64, api_key_env='DIAG_TEST_KEY', max_retries=0)
    provider = HttpEmbeddingProvider(config, transport=httpx.MockTransport(lambda request: httpx.Response(200,
        json={'output': {'embeddings': [{'text_index': 0, 'embedding': [10**400] * 64}]}})))
    token = d.begin('/search')
    try:
        with pytest.raises(EmbeddingFailure): provider.embed(['\ud800' if kind == 'unicode' else PRIVATE])
    finally:
        d.end(token)
    rows = lines(capsys)
    assert any(r.get('exception_type') == ('UnicodeEncodeError' if kind == 'unicode' else 'OverflowError') for r in rows)
    assert any(r.get('reason') == ('invalid_input_unicode' if kind == 'unicode' else 'invalid_vector') for r in rows)
