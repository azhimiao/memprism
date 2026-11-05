import contextlib
import http.client
import http.server
import json
from pathlib import Path
import socket
import sqlite3
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gateway import Gateway, BeforeDispatch


class Backend(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.server.calls.append((self.path, body, self.headers.get_all('Authorization')))
        count = len(self.server.calls)
        if self.server.commit is not None:
            value = json.loads(body)
            with sqlite3.connect(self.server.commit) as db:
                db.execute('INSERT OR IGNORE INTO receipts VALUES (?,?)',
                           (value['request_id'], body.decode()))
        if count <= self.server.disconnects:
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        data = self.server.body
        self.send_response(self.server.status)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Content-Type', 'application/json')
        self.send_header('Retry-After', '7')
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST


@contextlib.contextmanager
def running(*, status=200, disconnects=0, recovered=lambda identity: True, identity=lambda: {'pid': 1},
            commit=None, max_body=10485760):
    backend = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Backend)
    backend.calls = []
    backend.status = status
    backend.body = b'{"data":[]}'
    backend.disconnects = disconnects
    backend.commit = commit
    events = []
    gateway = Gateway(('127.0.0.1', 0), backend.server_address, identity, recovered,
                      recovery_seconds=.21, max_body=max_body, emit=lambda **k: events.append(k))
    for server in (backend, gateway):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield gateway, backend, events
    finally:
        for server in (gateway, backend):
            server.shutdown()
            server.server_close()


def request(gateway, path='/search', body=b'{"query":"non-sensitive synthetic"}', method='POST'):
    client = http.client.HTTPConnection(*gateway.server_address, timeout=3)
    try:
        client.request(method, path, body=body, headers={'Content-Type': 'application/json',
                                                       'Authorization': 'Bearer synthetic'})
        response = client.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        client.close()


@pytest.mark.parametrize('status', [200, 401, 403, 404, 422, 429, 500, 502, 503])
def test_http_business_status_never_retried(status):
    with running(status=status) as (gateway, backend, events):
        result, body, headers = request(gateway)
        assert result == status and body == backend.body
        assert headers['Retry-After'] == '7'
        assert len(backend.calls) == 1 and not events


def test_disconnect_once_then_exact_body_and_headers_replayed():
    with running(disconnects=1) as (gateway, backend, events):
        assert request(gateway)[0] == 200
        assert len(backend.calls) == 2 and backend.calls[0] == backend.calls[1]
        assert events[0]['event'] == 'transport_replay'


def test_persistent_disconnect_at_most_two_attempts():
    with running(disconnects=10) as (gateway, backend, events):
        assert request(gateway)[0] == 503
        assert len(backend.calls) == 2
        assert events[-1]['event'] == 'transport_replay_failed'


@pytest.mark.parametrize('identity', [lambda: None, lambda: {'pid': 1}])
def test_live_unknown_or_not_recovered_worker_cannot_authorize_replay(identity):
    with running(disconnects=1, recovered=lambda original: False, identity=identity) as (gateway, backend, events):
        assert request(gateway)[0] == 503
        assert len(backend.calls) == 1 and not events


@pytest.mark.parametrize('body', [b'{}', b'[]', b'not json', b'{"request_id":"x"}'])
def test_add_requires_stable_existing_identities_for_replay(body):
    with running(disconnects=1) as (gateway, backend, events):
        assert request(gateway, '/add', body)[0] == 503
        assert len(backend.calls) == 1 and not events


def test_add_reply_lost_after_commit_replays_same_receipt_without_duplicate(tmp_path):
    path = tmp_path/'receipts.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE receipts (request_id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    body = json.dumps(dict(request_id='r', user_id='u', session_id='s', messages=[{'content': 'synthetic'}])).encode()
    with running(disconnects=1, commit=path) as (gateway, backend, events):
        assert request(gateway, '/add', body)[0] == 200
        assert len(backend.calls) == 2 and backend.calls[0] == backend.calls[1]
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM receipts').fetchone()[0] == 1


def test_body_limit_is_enforced_before_backend():
    with running(max_body=3) as (gateway, backend, events):
        assert request(gateway, body=b'abcd')[0] == 413
        assert not backend.calls


def test_no_unexpected_route_replay():
    with running(disconnects=1) as (gateway, backend, events):
        assert request(gateway, '/other')[0] == 503
        assert len(backend.calls) == 1


def test_transfer_encoding_is_rejected_before_backend():
    with running() as (gateway, backend, events):
        client = http.client.HTTPConnection(*gateway.server_address)
        client.request('POST', '/search', body=b'0\r\n\r\n', headers={'Transfer-Encoding': 'chunked'})
        response = client.getresponse()
        assert response.status == 400
        response.read()
        client.close()
        assert not backend.calls


def test_refused_before_dispatch_can_wait_for_same_starting_worker(monkeypatch):
    from gateway import Handler
    original = Handler.exchange
    attempted = []

    def exchange(self, body):
        attempted.append(body)
        if len(attempted) == 1:
            raise BeforeDispatch()
        return original(self, body)

    monkeypatch.setattr(Handler, 'exchange', exchange)
    with running(recovered=lambda old: False) as (gateway, backend, events):
        gateway.ready = lambda: True
        assert request(gateway)[0] == 200
        assert len(attempted) == 2 and len(backend.calls) == 1
        assert events[0]['event'] == 'transport_wait_before_dispatch'


def test_sixteen_concurrent_requests_preserve_bodies():
    from concurrent.futures import ThreadPoolExecutor
    with running() as (gateway, backend, events):
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda i: request(gateway, body=json.dumps({'query': str(i)}).encode())[0], range(32)))
        assert results == [200] * 32
        assert len(backend.calls) == 32 and len({body for _, body, _ in backend.calls}) == 32


def test_incomplete_success_response_is_not_sent_to_client(monkeypatch):
    from gateway import Handler
    original = Handler.exchange
    count = []

    def exchange(self, body):
        count.append(1)
        if len(count) == 1:
            raise http.client.IncompleteRead(b'{"partial":')
        return original(self, body)

    monkeypatch.setattr(Handler, 'exchange', exchange)
    with running() as (gateway, backend, events):
        status, body, headers = request(gateway)
        assert status == 200 and body == backend.body and b'partial' not in body
        assert len(count) == 2
