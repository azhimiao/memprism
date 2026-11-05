"""Measure retained Python allocation for actual SQLite readiness inputs."""
from contextlib import closing
import gc
import importlib.util
import json
from pathlib import Path
import sqlite3
import tracemalloc

from app import segments
from app.api import create_app
from app.config import Settings
from app.embeddings import EmbeddingBatch, EmbeddingConfig
from app.models import AddRequest


class Provider:
    implementation_kind = 'engineering_double'
    config = EmbeddingConfig(provider='engineering', endpoint='https://invalid.test/embeddings', dimensions=2)
    def embed(self, texts):
        return EmbeddingBatch([[1., 0.] for _ in texts])


def test_vector_history_is_streamed_without_weakening_actual_byte_validation(tmp_path):
    app = create_app(Settings(db_path=tmp_path/'memory.db', profile_id='cycle2_dense'),
                     embedding_provider=Provider())
    app.state.ingest.add(AddRequest(user_id='u', session_id='s', request_id='r',
        messages=[{'role':'user', 'content':'alpha source ' + str(i)} for i in range(128)]))
    # Valid JSON with intentionally large whitespace isolates storage residency
    # from tokenizer/model behaviour; 128 independent 32KiB vector values.
    vector = '[' + ' ' * 32768 + '1.0,0.0]'
    with sqlite3.connect(app.state.store.db_path) as db:
        db.execute('UPDATE span_vectors SET vector_json=?', (vector,))
    path = Path(__file__).parents[2]/'experiments/073-lease-memory-root-relief/segments.py'
    spec = importlib.util.spec_from_file_location('segments071_memory_baseline', path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    def measure(module):
        with module._counts_lock:
            module._counts_cache.clear()
            module._source_counts_cache.clear()
            module._counts_cache_bytes = module._source_counts_cache_bytes = 0
        gc.collect()
        with closing(app.state.store._connect()) as db:
            db.execute('BEGIN')
            tracemalloc.start()
            try:
                result = module.counts(db, 'u', app.state.profile)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        return result, peak
    old, old_peak = measure(baseline)
    new, new_peak = measure(segments)
    assert old == new == dict.fromkeys(('active', 'raw', 'span', 'fts', 'dense'), 128)
    assert new_peak < old_peak * .5, (old_peak, new_peak)
    print(json.dumps({'scope':'synthetic Python allocation; not production RSS',
                      'baseline_peak_bytes':old_peak, 'streamed_peak_bytes':new_peak}))
