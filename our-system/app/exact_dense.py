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


def exact_top_scores(serialized, query_vector, dimensions, limit, tie_keys):
    if len(serialized) != len(tie_keys) or limit < 1 or dimensions < 1:
        raise ValueError('invalid_exact_dense_input')
    query = normalize_vector(query_vector, dimensions)
    if not serialized:
        return []
    try:
        vectors = [_vector_parser.validate_json(s) for s in serialized]
    except (ValidationError, TypeError):
        raise EmbeddingFailure('invalid_vector') from None
    if any(len(v) != dimensions for v in vectors):
        raise EmbeddingFailure('dimension_mismatch')
    matrix = np.asarray(vectors, dtype=np.float64)
    scales = np.max(np.abs(matrix), axis=1)
    if not np.isfinite(matrix).all() or np.any(scales == 0):
        raise EmbeddingFailure('invalid_vector')
    # Scale first, like normalize_vector, to handle extreme finite components.
    scaled = matrix / scales[:, None]
    norms = np.sqrt(np.sum(scaled * scaled, axis=1))
    units = scaled / norms[:, None]
    approximate = np.sum(units * np.asarray(query), axis=1)
    # Positive scaled norms and sums of unit-vector products have bounded
    # absolute error. This dominates normalization plus dot-product roundoff;
    # fall back to all rows if dimension invalidates the envelope assumptions.
    envelope = 128 * dimensions * sys.float_info.epsilon
    k = min(limit, len(vectors))
    if envelope >= .01:
        indices = range(len(vectors))
    else:
        boundary = float(np.partition(approximate, len(vectors)-k)[len(vectors)-k])
        indices = np.flatnonzero(approximate >= boundary - 2 * envelope)
    exact = []
    for i in indices:
        i = int(i)
        vector = normalize_vector(vectors[i], dimensions)
        score = max(-1., min(1., math.fsum(a*b for a,b in zip(query, vector))))
        exact.append((i, score))
    exact.sort(key=lambda item: (-item[1], *tie_keys[item[0]]))
    return exact[:k]
