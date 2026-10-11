"""Small ASGI guard: explicit API authentication, bounded bodies and concurrency."""
from __future__ import annotations
import hmac
import os
import threading
from starlette.responses import JSONResponse
from app.config import Settings


class TransportGuard:
    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings
        self.key = os.environ.get(settings.service_key_env, "").encode("utf-8")
        self.slots = threading.BoundedSemaphore(settings.max_concurrent_requests)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in {"/add","/search"}:
            return await self.app(scope,receive,send)
        mode = self.settings.auth_mode
        if mode != "none":
            header_name = b"x-api-key" if mode == "x-api-key" else b"authorization"
            values = [v for k,v in scope.get("headers",[]) if k.lower() == header_name]
            expected = self.key if mode == "x-api-key" else (b"Bearer " if mode == "bearer" else b"Token ") + self.key
            if len(values) != 1 or not hmac.compare_digest(values[0], expected):
                return await JSONResponse({"detail":"unauthorized"},status_code=401)(scope,receive,send)
        if not self.slots.acquire(blocking=False):
            return await JSONResponse({"detail":"service capacity reached"},status_code=503,
                                      headers={"Retry-After":"1"})(scope,receive,send)
        try:
            events, size = [], 0
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    return
                if event["type"] != "http.request":
                    continue
                size += len(event.get("body",b""))
                if size > self.settings.max_request_bytes:
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
            await self.app(scope,replay_receive,send)
        finally:
            self.slots.release()
