"""Transient float64 screening; every returned score uses the original fsum.

No vector/text cache or ANN. All rows are hashed on every search; exact-hash validation proofs are bounded. A generous
roundoff envelope retains possible winners including ties for exact reranking.
The caller owns the user-local SQLite snapshot, source checks and epoch fence.
"""
from __future__ import annotations

import math
import sys
import numpy as np
from app.embeddings import normalize_vector, EmbeddingFailure
from app.vector_checks import decode_validated_vector

# At most ~12 MiB of float64 matrices per search (six matrix temporaries),
# plus the Python parsed block. Never retain vectors beyond this invocation.
_MAX_BLOCK_COMPONENTS = 262144


def _scalar_score(vector, query, dimensions):
    unit = normalize_vector(vector, dimensions)
    return max(-1., min(1., math.fsum(a*b for a,b in zip(query, unit))))


def exact_top_scores(serialized, query_vector, dimensions, limit, tie_keys):
    if len(serialized) != len(tie_keys) or limit < 1 or dimensions < 1:
        raise ValueError('invalid_exact_dense_input')
    query = normalize_vector(query_vector, dimensions)
    if not serialized:
        return []
    k = min(limit, len(serialized))
    block_rows = max(1, _MAX_BLOCK_COMPONENTS // dimensions)
    order = lambda item: (-item[1], *tie_keys[item[0]])
    # Keep only one float64 score per row, never a corpus-sized vector matrix.
    # Validate EVERY block before selecting global possible winners. This avoids
    # recomputing an exact top-k independently in every block.
    approximate_scores = np.empty(len(serialized), dtype=np.float64)
    envelope = 128 * dimensions * sys.float_info.epsilon
    scalar_all = dimensions > _MAX_BLOCK_COMPONENTS or envelope >= .01
    for start in range(0, len(serialized), block_rows):
        try:
            vectors = [decode_validated_vector(s, dimensions) for s in serialized[start:start+block_rows]]
        except (ValueError, TypeError, OverflowError, RecursionError):
            raise EmbeddingFailure('invalid_vector') from None
        if any(len(v) != dimensions for v in vectors):
            raise EmbeddingFailure('dimension_mismatch')
        # A single huge vector takes O(dimensions), never O(corpus*dimensions).
        # Every block is validated, even if none of its rows can win.
        if scalar_all:
            for i, vector in enumerate(vectors):
                approximate_scores[start+i] = _scalar_score(vector, query, dimensions)
        else:
            matrix = np.asarray(vectors, dtype=np.float64)
            scales = np.max(np.abs(matrix), axis=1)
            if not np.isfinite(matrix).all() or np.any(scales == 0):
                raise EmbeddingFailure('invalid_vector')
            scaled = matrix / scales[:, None]
            norms = np.sqrt(np.sum(scaled * scaled, axis=1))
            units = scaled / norms[:, None]
            approximate = np.sum(units * np.asarray(query), axis=1)
            approximate_scores[start:start+len(vectors)] = approximate
    if scalar_all:
        return sorted(enumerate(approximate_scores.tolist()), key=order)[:k]
    boundary = float(np.partition(approximate_scores, len(serialized)-k)[len(serialized)-k])
    # Same conservative bound as the blockwise version. Any exact global winner
    # must be within 2*envelope of the approximate kth score, including ties.
    indices = np.flatnonzero(approximate_scores >= boundary - 2 * envelope)
    exact = []
    for index in indices:
        i = int(index)
        vector = decode_validated_vector(serialized[i], dimensions)
        exact.append((i, _scalar_score(vector, query, dimensions)))
    return sorted(exact, key=order)[:k]
