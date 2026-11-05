"""Reuse TCP/TLS connections only; client cookies/headers remain per embed call."""
import threading
import httpx

class _Lease(httpx.BaseTransport):
    def __init__(self,owner):self.owner=owner
    def handle_request(self,request):return self.owner.transport().handle_request(request)
    def close(self):pass  # The provider owner, never a single request, owns the pool.

class TransportPool:
    def __init__(self,transport=None):
        self._lock=threading.Lock()
        self._transport=transport
        self._closed=False
    def transport(self):
        with self._lock:
            if self._closed:raise RuntimeError('embedding_transport_closed')
            if self._transport is None:
                self._transport=httpx.HTTPTransport(trust_env=False,retries=0,
                    limits=httpx.Limits(max_connections=32,max_keepalive_connections=16,keepalive_expiry=30))
            return self._transport
    def lease(self):return _Lease(self)
    def close(self):
        with self._lock:
            if not self._closed:
                self._closed=True
                if self._transport is not None:self._transport.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
