"""Keep real SQLite source-vector validation bounded by the source, not user size."""
import importlib.util
from pathlib import Path
import sqlite3

import pytest

from app import segments


@pytest.mark.parametrize('analyzed', [False, True])
def test_source_vector_lookup_scales_and_preserves_filters(analyzed):
    with sqlite3.connect(':memory:') as db:
        segments.initialize(db)
        for user, count in [('u', 4096), ('other', 32)]:
            db.executemany('INSERT INTO source_spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                [(f'{user}-{i:05}', user, f'raw-{i}', 1, 'h', 0, 1, 'h', 'f', 'x', 0,
                  'complete', 1, None, None) for i in range(count)])
            db.executemany('INSERT INTO span_vectors VALUES (?,?,?,?,?,?)',
                [(user, f'{user}-{i:05}', 'model', 2, 'h', '[1,0]') for i in range(count)])
        # Same span with another model/dimensions must not enter this read.
        db.execute("INSERT INTO span_vectors VALUES ('u','u-00000','other-model',3,'h','[1,0,0]')")
        if analyzed:
            db.execute('ANALYZE')
        args = ('u', 'raw-0', 'model', 2)
        plan = [r[3] for r in db.execute('EXPLAIN QUERY PLAN '+segments._SOURCE_VECTOR_SQL, args)]
        steps = 0
        def tick():
            nonlocal steps
            steps += 1
            return int(steps > 1500)
        db.set_progress_handler(tick, 1)
        try:
            result = db.execute(segments._SOURCE_VECTOR_SQL, args).fetchall()
        finally:
            db.set_progress_handler(None, 0)
        assert result == [('u', 'u-00000', 'model', 2, 'h', '[1,0]')]
        assert steps < 1500, (steps, plan)
        # This is the exact previous release's query, recovered from its code,
        # not a reconstructed baseline or an assertion about source spelling.
        path = Path(__file__).parents[2]/'experiments/073-lease-memory-root-relief/bundle-01/payload/segments.py'
        spec = importlib.util.spec_from_file_location('segments073_query_baseline', path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        sql = next(v for v in baseline.counts.__code__.co_consts
                   if isinstance(v, str) and 'SELECT v.* FROM source_spans s' in v)
        assert db.execute(sql, args).fetchall() == result
        assert db.execute(segments._SOURCE_VECTOR_SQL, ('u', 'missing', 'model', 2)).fetchall() == []
        assert db.execute(segments._SOURCE_VECTOR_SQL, ('other', 'raw-0', 'model', 2)).fetchall()[0][0] == 'other'
