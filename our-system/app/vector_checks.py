"""Cache only validation booleans keyed by full vector JSON hash and dimension.

Readiness still reads and hashes every persisted vector on every check. Neither
vectors nor text survive in this cache after source erasure; mutations produce
a new hash and must pass numerical validation again.
"""
from collections import OrderedDict
import hashlib
import json
import threading
import sys
from app.embeddings import normalize_vector,EmbeddingFailure
from app.vector_json import parse_vector_json

_cache=OrderedDict()
_lock=threading.RLock()
_cache_sizes={}
_cache_bytes=0
_CACHE_LIMIT=65536
_CACHE_MAX_BYTES=16*1024*1024


def _key(serialized,dimensions):
    return (dimensions,hashlib.sha256(serialized.encode("utf-8")).hexdigest())


def _lookup(key):
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None


def _remember(key,valid):
    global _cache_bytes
    with _lock:
        size=sys.getsizeof(key)+sum(sys.getsizeof(x) for x in key)+sys.getsizeof(valid)+384
        if key in _cache:
            _cache.pop(key)
            _cache_bytes-=_cache_sizes.pop(key)
        if size<=_CACHE_MAX_BYTES:
            _cache[key]=valid
            _cache_sizes[key]=size
            _cache_bytes+=size
        while len(_cache)>_CACHE_LIMIT or _cache_bytes>_CACHE_MAX_BYTES:
            removed,_=_cache.popitem(last=False)
            _cache_bytes-=_cache_sizes.pop(removed)


def decode_validated_vector(serialized,dimensions):
    """Decode fresh values; reuse only an exact-hash numerical validation proof.

    A True entry proves finite numeric components, matching dimension, nonzero.
    No parsed values survive this invocation. False entries use the strict path
    again, preserving the caller's dimension/zero-vector failure semantics.
    """
    key=_key(serialized,dimensions)
    if _lookup(key) is True:
        return json.loads(serialized)
    try:
        values=parse_vector_json(serialized)
    except (ValueError,TypeError,OverflowError,RecursionError):
        _remember(key,False)
        raise
    _remember(key,len(values)==dimensions and bool(values) and any(values))
    return values


def valid_vector_json(serialized,dimensions):
    if not isinstance(serialized,str):return False
    key=_key(serialized,dimensions)
    saved=_lookup(key)
    if saved is not None:return saved
    try:
        values=parse_vector_json(serialized)
        valid=len(values)==dimensions and bool(values) and any(values)
    except (EmbeddingFailure,ValueError,TypeError,OverflowError,RecursionError):
        valid=False
    _remember(key,valid)
    return valid
