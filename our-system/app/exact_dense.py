"""Transient float64 screening; every returned score uses the original fsum.

No vector/text cache or ANN. All rows are validated on every search. A generous
roundoff envelope retains possible winners including ties for exact reranking.
The caller owns the user-local SQLite snapshot, source checks and epoch fence.
"""
from __future__ import annotations

import math
import sys
import numpy as np
from pydantic_core import SchemaValidator, core_schema, ValidationError
from app.embeddings import normalize_vector, EmbeddingFailure

_vector_parser = SchemaValidator(core_schema.list_schema(
    core_schema.float_schema(strict=True, allow_inf_nan=False)))

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
    exact = []
    k = min(limit, len(serialized))
    block_rows = max(1, _MAX_BLOCK_COMPONENTS // dimensions)
    order = lambda item: (-item[1], *tie_keys[item[0]])
    for start in range(0, len(serialized), block_rows):
        try:
            vectors = [_vector_parser.validate_json(s) for s in serialized[start:start+block_rows]]
        except (ValidationError, TypeError):
            raise EmbeddingFailure('invalid_vector') from None
        if any(len(v) != dimensions for v in vectors):
            raise EmbeddingFailure('dimension_mismatch')
        # A single huge vector takes O(dimensions), never O(corpus*dimensions).
        # Every block is validated, even if none of its rows can win.
        if dimensions > _MAX_BLOCK_COMPONENTS:
            indices = range(len(vectors))
        else:
            matrix = np.asarray(vectors, dtype=np.float64)
            scales = np.max(np.abs(matrix), axis=1)
            if not np.isfinite(matrix).all() or np.any(scales == 0):
                raise EmbeddingFailure('invalid_vector')
            scaled = matrix / scales[:, None]
            norms = np.sqrt(np.sum(scaled * scaled, axis=1))
            units = scaled / norms[:, None]
            approximate = np.sum(units * np.asarray(query), axis=1)
            envelope = 128 * dimensions * sys.float_info.epsilon
            local_k = min(k, len(vectors))
            if envelope >= .01:
                indices = range(len(vectors))
            else:
                boundary = float(np.partition(approximate, len(vectors)-local_k)[len(vectors)-local_k])
                # A block's top k contains every possible global winner. Include
                # the complete roundoff/tie envelope before exact reranking.
                threshold = boundary - 2 * envelope
                if len(exact) == k:
                    threshold = max(threshold, exact[-1][1] - 2 * envelope)
                indices = np.flatnonzero(approximate >= threshold)
        for i in indices:
            i = int(i)
            exact.append((start+i, _scalar_score(vectors[i], query, dimensions)))
        exact.sort(key=order)
        del exact[k:]
    return exact
