"""Loopback recovery gateway; only replay after a proven owned-worker exit.

No cache, persisted request body, successful placeholder, or status-code retry.
The backend remains the sole authority for authentication, Add receipts and Search.
"""
import http.client
import http.server
import json
import socket
import threading
import time

HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
       'te', 'trailer', 'transfer-encoding', 'upgrade', 'content-length'}


class BeforeDispatch(OSError):
    """The TCP connection was refused before any request bytes were sent."""


class Gateway(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    request_queue_size = 64

    def __init__(self, address, backend, identity, recovered, *, max_body=10485760,
                 recovery_seconds=75, transport_timeout=180, ready=lambda: False, emit=lambda **k: None):
        self.backend = backend
        self.identity = identity
        self.recovered = recovered
        self.ready = ready
        self.max_body = max_body
        self.recovery_seconds = recovery_seconds
        self.transport_timeout = transport_timeout
        self.emit = emit
        self.slots = threading.BoundedSemaphore(32)
        super().__init__(address, Handler)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def reply(self, status, body, headers=()):
        self.send_response_only(status)
        for key, value in headers:
            if key.lower() not in HOP:
                self.send_header(key, value)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def unavailable(self, reason, status=503):
        self.reply(status, json.dumps({'detail': reason}).encode(),
                   [('Content-Type', 'application/json'), ('Retry-After', '1')])

    def exchange(self, body):
        connection = http.client.HTTPConnection(*self.server.backend,
                                                timeout=self.server.transport_timeout)
        try:
            try:
                connection.connect()
            except ConnectionRefusedError as error:
                raise BeforeDispatch() from error
            connection.putrequest(self.command, self.path, skip_host=True,
                                  skip_accept_encoding=True)
            excluded = HOP | {x.strip().lower() for x in self.headers.get('Connection', '').split(',')}
            for key, value in self.headers.items():
                if key.lower() not in excluded:
                    connection.putheader(key, value)
            connection.putheader('Content-Length', str(len(body)))
            connection.putheader('Connection', 'close')
            connection.endheaders(body)
            response = connection.getresponse()
            data = response.read(64 * 1024**2 + 1)
            if len(data) > 64 * 1024**2:
                raise ValueError('response_limit')
            if response.length not in (None, 0):
                raise http.client.IncompleteRead(data)
            return response.status, data, response.getheaders()
        finally:
            connection.close()

    def replay_safe(self, body):
        if self.command != 'POST':
            return False
        if self.path == '/search':
            return True
        if self.path != '/add':
            return False
        try:
            value = json.loads(body)
            return isinstance(value, dict) and all(isinstance(value.get(k), str) and value[k].strip()
                for k in ('request_id', 'user_id', 'session_id'))
        except (ValueError, TypeError, UnicodeDecodeError):
            return False

    def handle_request(self):
        sizes = self.headers.get_all('Content-Length') or []
        if self.headers.get('Transfer-Encoding') or len(sizes) > 1:
            return self.unavailable('unsupported framing', 400)
        try:
            length = int(sizes[0]) if sizes else 0
        except ValueError:
            return self.unavailable('invalid length', 400)
        if length < 0:
            return self.unavailable('invalid length', 400)
        if length > self.server.max_body:
            return self.unavailable('request too large', 413)
        body = self.rfile.read(length)
        if len(body) != length:
            return self.unavailable('incomplete body', 400)
        original = self.server.identity()
        try:
            status, data, headers = self.exchange(body)
        except BeforeDispatch:
            # Restart may already have published the new PID before it listens.
            # No bytes reached that worker, so wait then dispatch exactly once.
            deadline = time.monotonic() + self.server.recovery_seconds
            while time.monotonic() < deadline:
                if self.server.ready():
                    self.server.emit(event='transport_wait_before_dispatch', endpoint=self.path)
                    try:
                        status, data, headers = self.exchange(body)
                    except (OSError, http.client.HTTPException, ValueError):
                        return self.unavailable('backend recovery unsuccessful')
                    break
                time.sleep(.2)
            else:
                return self.unavailable('backend readiness deadline')
        except (OSError, http.client.HTTPException) as error:
            # A 4xx/5xx HTTP response never enters this branch. A timeout alone
            # does not authorize replay while the original worker remains live.
            if original is None or not self.replay_safe(body):
                return self.unavailable('backend transport unavailable')
            deadline = time.monotonic() + self.server.recovery_seconds
            while time.monotonic() < deadline:
                if self.server.recovered(original):
                    self.server.emit(event='transport_replay', endpoint=self.path,
                                     first_error=type(error).__name__, max_attempts=2)
                    try:
                        status, data, headers = self.exchange(body)
                    except (OSError, http.client.HTTPException, ValueError) as second:
                        self.server.emit(event='transport_replay_failed', endpoint=self.path,
                                         error_type=type(second).__name__)
                        return self.unavailable('backend recovery unsuccessful')
                    break
                time.sleep(.2)
            else:
                return self.unavailable('backend recovery deadline')
        except ValueError:
            return self.unavailable('backend response limit')
        self.reply(status, data, headers)

    def dispatch(self):
        if not self.server.slots.acquire(blocking=False):
            return self.unavailable('gateway capacity reached')
        try:
            self.handle_request()
        except (OSError, http.client.HTTPException):
            self.close_connection = True
        finally:
            self.server.slots.release()

    do_POST = dispatch
    do_GET = dispatch
    do_HEAD = dispatch
    do_OPTIONS = dispatch
