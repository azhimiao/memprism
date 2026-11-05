"""Strict persisted-vector parsing without the failing native schema hot loop.

Scores still use float64 screening plus original scalar fsum. This parser only
changes how JSON numbers are validated; no persisted vector or result is cached.
"""
import json
import math


def parse_vector_json(serialized):
    values = json.loads(serialized)
    if not isinstance(values, list):
        raise ValueError('invalid_vector')
    for i, value in enumerate(values):
        if type(value) not in (int, float):
            raise ValueError('invalid_vector')
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('invalid_vector')
        values[i] = number
    return values
