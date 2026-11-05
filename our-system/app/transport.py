"""Small ASGI guard: explicit API authentication, bounded bodies and concurrency."""
from __future__ import annotations
import hmac
import os
import threading
import time
import sys
from starlette.responses import JSONResponse
from app.config import Settings
from app import diagnostics as diagnostic


class TransportGuard:
    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings
        self.key = os.environ.get(settings.service_key_env, "").encode("utf-8")
        self.slots = threading.BoundedSemaphore(settings.max_concurrent_requests)
        self.active_lock = threading.Lock()
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in {"/add","/search"}:
            return await self.app(scope,receive,send)
        token = diagnostic.begin(scope['path'])
        started = time.perf_counter()
        status, complete, retry_after = None, False, None
        response_bytes = 0
        async def observed_send(event):
            nonlocal status, complete, retry_after, response_bytes
            if event['type'] == 'http.response.start':
                status = event['status']
                for key, value in event.get('headers', []):
                    if key.lower() == b'retry-after' and len(value) <= 8 and value.isdigit():
                        retry_after = int(value)
                event = dict(event, headers=list(event.get('headers', [])) +
                             [(b'x-aml-diagnostic-id', diagnostic.current_id().encode('ascii'))])
            await send(event)
            if event["type"] == "http.response.body": response_bytes += len(event.get("body", b""))
            if event['type'] == 'http.response.body' and not event.get('more_body', False): complete = True
        diagnostic.emit('request_started')
        try:
            await self._handle_http(scope, receive, observed_send)
        except BaseException as exc:
            diagnostic.emit('request_failed', exc=exc)
            raise
        finally:
            diagnostic.flush_stage()
            diagnostic.emit('request_finished', status=status, response_complete=complete,
                            response_bytes=response_bytes,
                            retry_after_seconds=retry_after,
                            elapsed_ms=round((time.perf_counter() - started) * 1000, 3))
            diagnostic.end(token)

    async def _handle_http(self, scope, receive, send):
        diagnostic.stage('authentication')
        mode = self.settings.auth_mode
        if mode != "none":
            header_name = b"x-api-key" if mode == "x-api-key" else b"authorization"
            values = [v for k,v in scope.get("headers",[]) if k.lower() == header_name]
            expected = self.key if mode == "x-api-key" else (b"Bearer " if mode == "bearer" else b"Token ") + self.key
            if len(values) != 1 or not hmac.compare_digest(values[0], expected):
                diagnostic.reason('unauthorized')
                return await JSONResponse({"detail":"unauthorized"},status_code=401)(scope,receive,send)
        diagnostic.stage('capacity')
        if not self.slots.acquire(blocking=False):
            diagnostic.reason('service capacity reached')
            diagnostic.emit('input_shape', active=self.active, limit=self.settings.max_concurrent_requests,
                            rejection_type='capacity', exception_type='none',
                            branch_module=__name__, branch_line=sys._getframe().f_lineno)
            return await JSONResponse({"detail":"service capacity reached"},status_code=503,
                                      headers={"Retry-After":"1"})(scope,receive,send)
        try:
            with self.active_lock:
                self.active += 1
                active = self.active
            diagnostic.emit('input_shape', active=active, limit=self.settings.max_concurrent_requests)
            diagnostic.stage('request_body')
            events, size = [], 0
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    diagnostic.reason('client_disconnected')
                    return
                if event["type"] != "http.request":
                    continue
                size += len(event.get("body",b""))
                if size > self.settings.max_request_bytes:
                    diagnostic.reason('request body exceeds configured limit')
                    return await JSONResponse({"detail":"request body exceeds configured limit"},status_code=413)(scope,receive,send)
                events.append(event)
                if not event.get("more_body",False):
                    break
            iterator = iter(events)
            async def replay_receive():
                try:
                    return next(iterator)
                except StopIteration:
                    return await receive()
            diagnostic.emit('input_shape', body_bytes=size)
            diagnostic.stage('validation_and_dispatch')
            await self.app(scope,replay_receive,send)
        finally:
            with self.active_lock: self.active -= 1
            self.slots.release()
