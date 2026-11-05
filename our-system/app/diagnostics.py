"""Bounded structured diagnostics, excluding text, credentials and exception messages."""
from __future__ import annotations
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import json
import re
import sys
import threading
import time
from uuid import uuid4

VERSION = 'service-diagnostics-057-v1'
_LOCK = threading.Lock()
_STAGES = frozenset(('transport', 'authentication', 'capacity', 'request_body',
    'validation_and_dispatch', 'add', 'search', 'readiness_before_search', 'query_parse',
    'vector_coverage', 'readiness_before_embedding', 'embedding', 'readiness_after_embedding',
    'search_snapshot', 'evidence_selection', 'evidence_final_check', 'snapshot_retry'))
_EVENTS = frozenset(('request_started', 'request_finished', 'request_failed', 'operation_failed',
    'stage_started', 'stage_finished', 'stage_failed', 'stage_timing', 'embedding_http',
    'embedding_transport_failed', 'snapshot_retry', 'input_shape', 'startup'))
_REASONS = frozenset(('incomplete_catalog_route', 'unauthorized', 'service capacity reached',
    'request body exceeds configured limit', 'client_disconnected', 'control_not_ready',
    'search_failed', 'preparation_failed', 'lifecycle snapshot busy', 'embedding_timeout',
    'embedding_transport_error', 'embedding_rate_limited', 'embedding_provider_error',
    'embedding_provider_unavailable', 'invalid_embedding_response', 'input_too_long',
    'invalid_input', 'invalid_input_unicode', 'invalid_text_type', 'missing_api_key',
    'invalid_api_key', 'dimension_mismatch', 'response_count_mismatch', 'response_model_mismatch',
    'missing_configuration', 'incomplete_current_version', 'query_config_generation_changed',
    'route_rebuild_required', 'config_changed', 'incomplete_raw_route', 'incomplete_fts_route',
    'incomplete_dense_route', 'query_vector_unavailable', 'first_add_during_query_preparation',
    'index_hit_budget_exceeded', 'search_snapshot_changed', 'search_source_changed',
    'preparation_busy', 'lease_lost', 'lease_renewal_failed', 'required_embedding_not_ready',
    'request_payload_conflict'))
_REASONS = _REASONS | {'invalid_vector', 'zero_vector', 'invalid_response_index',
    'query_context_budget_exceeded', 'query_window_budget_exceeded',
    'catalog_read_reopened', 'readiness_audit_busy', 'readiness_database_replaced', 'prepared_sources_changed',
    'prepared_spans_changed', 'required_atom_not_ready', 'health_snapshot_changed'}
_NUMBERS = frozenset(('status', 'elapsed_ms', 'attempt', 'query_chars', 'query_utf8_bytes',
    'option_count', 'text_count', 'active', 'limit', 'body_bytes', 'max_text_bytes', 'sqlite_errorcode', 'branch_line',
    'retry_after_seconds', 'response_bytes', 'returned_items', 'output_characters',
    'packed_record_tokens', 'active_rows', 'snapshot_attempts'))
_MODULES = frozenset('app.' + name for name in ('api', 'transport', 'diagnostics', 'retrieval',
    'embeddings', 'embedding_accounting', 'jobs', 'store', 'memory_core', 'segments', 'exact_dense', 'vector_json', 'vector_checks',
    'queryplan', 'query', 'ingest', 'governance', 'reranking', 'profiles', 'models', 'atomic_memory',
    'evidence_navigation', 'capability_views', 'claim_projection', 'k0_migrate', 'set_selection',
    'evidence_budget', 'evidence_units', 'read_readiness', 'read_pool', 'write_runtime', 'evidence_catalog'))
_CLASSES = frozenset(('ValueError', 'TypeError', 'RuntimeError', 'KeyError', 'IndexError',
    'AttributeError', 'AssertionError', 'OSError', 'TimeoutError', 'MemoryError', 'OperationalError',
    'IntegrityError', 'DatabaseError', 'ProgrammingError', 'ReadTimeout', 'ConnectTimeout',
    'WriteTimeout', 'PoolTimeout', 'ConnectError', 'ReadError', 'WriteError', 'RemoteProtocolError',
    'LocalProtocolError', 'EmbeddingFailure', 'NotReady', 'SnapshotChanged', 'LifecycleSnapshotBusy',
    'RequestPayloadConflict', 'CancelledError', 'JSONDecodeError', 'OverflowError', 'UnicodeEncodeError',
    'InvalidSearchInput', 'SystemError', 'NameError', 'UnboundLocalError',
    'RecursionError', 'ReferenceError', 'ZeroDivisionError', 'FloatingPointError'))


@dataclass
class Diagnostic:
    request_id: str
    endpoint: str
    stage: str = 'transport'
    reason: str | None = None
    stage_started: float = field(default_factory=time.perf_counter)


_REQUEST: ContextVar[Diagnostic | None] = ContextVar('aml_diagnostic', default=None)


def safe_reason(value):
    if isinstance(value, str) and (value in _REASONS or re.fullmatch(r'embedding_http_[45][0-9][0-9]', value)):
        return value
    return 'unrecognized_reason'


def _exception_fields(exc):
    result = {'exception_type': type(exc).__name__ if type(exc).__name__ in _CLASSES else 'other_exception'}
    try:
        sqlite_code = getattr(exc, 'sqlite_errorcode', None)
    except Exception:
        sqlite_code = None
    if type(sqlite_code) is int: result['sqlite_errorcode'] = sqlite_code
    frames, tb = [], exc.__traceback__
    for _ in range(64):
        if tb is None: break
        module = tb.tb_frame.f_globals.get('__name__')
        if module in _MODULES:
            frames.append({'module': module, 'line': tb.tb_lineno})
        tb = tb.tb_next
    result['frames'] = frames[-12:]
    return result


def begin(endpoint):
    return _REQUEST.set(Diagnostic(uuid4().hex, endpoint if endpoint in ('/add', '/search') else 'other'))


def end(token):
    _REQUEST.reset(token)


def current_id():
    state = _REQUEST.get()
    return state.request_id if state else None


def stage(value):
    state = _REQUEST.get()
    if state is not None and value in _STAGES and value != state.stage:
        flush_stage()
        state.stage = value
        state.stage_started = time.perf_counter()


def flush_stage():
    state = _REQUEST.get()
    if state is not None:
        now = time.perf_counter()
        emit('stage_timing', elapsed_ms=round((now-state.stage_started)*1000,3))
        state.stage_started = now


def reason(value):
    state = _REQUEST.get()
    if state is not None: state.reason = safe_reason(value)


def emit(event, *, exc=None, **fields):
    if event not in _EVENTS: return
    try:
        state = _REQUEST.get()
        if state is None and event != 'startup': return
        row = {'version': VERSION, 'event': event,
               'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        if state is not None:
            row.update(request_id=state.request_id, endpoint=state.endpoint, stage=state.stage)
            if state.reason is not None: row['reason'] = state.reason
        for key, value in fields.items():
            if key in _NUMBERS and type(value) in (int, float) and abs(value) < 1e15:
                row[key] = value
            elif key == 'response_complete' and type(value) is bool: row[key] = value
            elif key == 'text_type' and value in ('document', 'query'): row[key] = value
            elif key == 'reason': row[key] = safe_reason(value)
            elif key == 'branch_module' and value in _MODULES: row[key] = value
            elif key == 'exception_type' and value == 'none': row[key] = value
            elif key == 'rejection_type' and value in ('authentication', 'capacity', 'body_limit', 'disconnect'):
                row[key] = value
        if exc is not None: row.update(_exception_fields(exc))
        line = 'AML_DIAG ' + json.dumps(row, ensure_ascii=True, separators=(',', ':'), allow_nan=False) + '\n'
        with _LOCK:
            sys.stderr.write(line)
            sys.stderr.flush()
    except Exception:
        # Logging failure must not turn a committed Add into an API failure.
        pass


def failure(exc, code=None):
    if code is not None: reason(code)
    emit('operation_failed', exc=exc)


def observe(value):
    if value not in _STAGES: raise ValueError('invalid diagnostic stage')
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            if _REQUEST.get() is None: return function(*args, **kwargs)
            stage(value)
            started = time.perf_counter()
            emit('stage_started')
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                try:
                    code = getattr(exc, 'code', None)
                except Exception:
                    code = None
                if isinstance(code, str): reason(code)
                emit('stage_failed', exc=exc, elapsed_ms=round((time.perf_counter() - started) * 1000, 3))
                raise
            emit('stage_finished', elapsed_ms=round((time.perf_counter() - started) * 1000, 3))
            return result
        return call
    return decorate
