"""Cache only validation booleans keyed by full vector JSON hash and dimension.

Readiness still reads and hashes every persisted vector on every check. Neither
vectors nor text survive in this cache after source erasure; mutations produce
a new hash and must pass numerical validation again.
"""
from collections import OrderedDict
import hashlib
import json
import threading

from app.embeddings import normalize_vector,EmbeddingFailure

_cache=OrderedDict()
_lock=threading.RLock()


def valid_vector_json(serialized,dimensions):
    if not isinstance(serialized,str):return False
    key=(dimensions,hashlib.sha256(serialized.encode("utf-8")).hexdigest())
    with _lock:
        if key in _cache:return _cache[key]
    try:
        normalize_vector(json.loads(serialized),dimensions)
        valid=True
    except (EmbeddingFailure,ValueError,TypeError):
        valid=False
    with _lock:
        _cache[key]=valid
        if len(_cache)>4096:_cache.popitem(last=False)
    return valid
