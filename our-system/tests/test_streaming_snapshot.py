"""Deterministic Streaming races; engineering vectors only, no network."""
from pathlib import Path
from dataclasses import replace

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.embeddings import EmbeddingBatch, EmbeddingConfig
from app.models import AddRequest
from app.jobs import NotReady, SnapshotChanged


class InterleavedProvider:
    implementation_kind = 'engineering_double'

    def __init__(self):
        self.config = EmbeddingConfig(provider='engineering', endpoint='https://invalid.test/embeddings', dimensions=2)
        self.query_calls = 0
        self.total_calls = 0
        self.query_action = None

    def embed(self, texts):
        self.total_calls += 1
        if texts == ['find target']:
            self.query_calls += 1
            if self.query_action:
                self.query_action()
        return EmbeddingBatch([[1., 0.] for _ in texts], {'requests': 1, 'reported_input_tokens': len(texts), 'usage_complete': True})


def write(app, content, request_id):
    return app.state.ingest.add(AddRequest(user_id='u', session_id='s', request_id=request_id,
        messages=[{'role': 'user', 'content': content}]))


def setup(path):
    provider = InterleavedProvider()
    app = create_app(Settings(db_path=path, profile_id='cycle2_dense', coordinator_lock_timeout_seconds=30),
        embedding_provider=provider)
    write(app, 'target old evidence', 'seed')
    return app, provider


def test_add_during_every_query_embedding_does_not_starve_search(tmp_path):
    app, provider = setup(tmp_path / 'stream.sqlite')
    provider.query_action = lambda: write(app, 'target latest evidence', 'during-query-' + str(provider.query_calls))
    with TestClient(app) as client:
        response = client.post('/search', json={'user_id': 'u', 'query': 'find target', 'top_k': 100})
    assert response.status_code == 200
    assert provider.query_calls == 1
    assert any(hit['content'] == 'target latest evidence' for hit in response.json()['data'])


def test_delete_during_query_embedding_is_applied_before_retrieval(tmp_path):
    app, provider = setup(tmp_path / 'delete.sqlite')
    provider.query_action = lambda: write(app, 'delete "target old evidence"', 'erase')
    with TestClient(app) as client:
        response = client.post('/search', json={'user_id': 'u', 'query': 'find target', 'top_k': 100})
    assert response.status_code == 200
    assert provider.query_calls == 1
    assert all(hit['content'] != 'target old evidence' for hit in response.json()['data'])


def test_add_success_is_visible_to_following_search(tmp_path):
    app, provider = setup(tmp_path / 'visible.sqlite')
    assert write(app, 'target published evidence', 'committed').success
    with TestClient(app) as client:
        response = client.post('/search', json={'user_id': 'u', 'query': 'find target', 'top_k': 100})
    assert response.status_code == 200
    assert any(hit['content'] == 'target published evidence' for hit in response.json()['data'])


def test_failed_publication_reuses_embedding_but_success_replay_does_no_work(tmp_path, monkeypatch):
    app, provider = setup(tmp_path / 'retry.sqlite')
    commit = app.state.store.commit_add
    attempts = 0
    def transient_failure(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise NotReady('injected_publication_failure')
        return commit(*args, **kwargs)
    monkeypatch.setattr(app.state.store, 'commit_add', transient_failure)
    payload = {'user_id': 'u', 'session_id': 's', 'request_id': 'retry',
        'messages': [{'role': 'user', 'content': 'target new retry evidence'}]}
    before = provider.total_calls
    with TestClient(app) as client:
        codes = [client.post('/add', json=payload).status_code for _ in range(3)]
    assert codes == [503, 200, 200]
    assert provider.total_calls - before == 1
    from app.embedding_accounting import summarize_ledger
    ledger = app.state.ingest.embedding_provider.ledger_path
    summary = summarize_ledger(ledger, price=.5, currency='CNY')
    assert summary['cache_hits'] == 1


def test_snapshot_retries_reuse_one_query_vector(tmp_path, monkeypatch):
    app, provider = setup(tmp_path / 'retry-query.sqlite')
    check = app.state.coordinator.final_check
    attempts = 0
    def changed_twice(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise SnapshotChanged('injected_snapshot_change')
        return check(*args, **kwargs)
    monkeypatch.setattr(app.state.coordinator, 'final_check', changed_twice)
    with TestClient(app) as client:
        response = client.post('/search', json={'user_id': 'u', 'query': 'find target', 'top_k': 100})
    assert response.status_code == 200
    assert attempts == 3 and provider.query_calls == 1


def test_first_add_between_empty_coverage_and_search_is_not_lost(tmp_path, monkeypatch):
    provider = InterleavedProvider()
    app = create_app(Settings(db_path=tmp_path / 'first.sqlite', profile_id='cycle2_dense'),
        embedding_provider=provider)
    coverage = app.state.store.vector_coverage
    injected = False
    def publish_first(*args, **kwargs):
        nonlocal injected
        result = coverage(*args, **kwargs)
        if not injected:
            injected = True
            assert result == (0, 0)
            assert write(app, 'target first evidence', 'first').success
        return result
    monkeypatch.setattr(app.state.store, 'vector_coverage', publish_first)
    with TestClient(app) as client:
        response = client.post('/search', json={'user_id': 'u', 'query': 'find target', 'top_k': 100})
    assert response.status_code == 200
    assert provider.query_calls == 1
    assert any(hit['content'] == 'target first evidence' for hit in response.json()['data'])


def test_config_aba_during_embedding_cannot_adopt_new_generation(tmp_path):
    from app.jobs import JobCoordinator
    from app.profiles import Profile
    app, provider = setup(tmp_path / 'aba.sqlite')
    def change_away_and_back():
        changed = replace(app.state.retrieval.settings, return_char_budget=12345)
        other = JobCoordinator(app.state.store, Profile.build(changed, app.state.ingest.embedding_provider))
        other.activate()
        app.state.coordinator.activate()
    provider.query_action = change_away_and_back
    with TestClient(app) as client:
        response = client.post('/search', json={'user_id': 'u', 'query': 'find target', 'top_k': 100})
    assert response.status_code == 503
    assert response.json()['detail'] == 'query_config_generation_changed'
    assert provider.query_calls == 1


def test_continuous_changes_remain_bounded_and_never_return_stale_evidence(tmp_path, monkeypatch):
    app, provider = setup(tmp_path / 'continuous.sqlite')
    attempts = 0
    def always_changed(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise SnapshotChanged('continuous_source_change')
    monkeypatch.setattr(app.state.coordinator, 'final_check', always_changed)
    with TestClient(app) as client:
        response = client.post('/search', json={'user_id': 'u', 'query': 'find target', 'top_k': 100})
    assert response.status_code == 503 and 'data' not in response.json()
    assert attempts == app.state.profile.identity['search_attempts'] == 8
    assert provider.query_calls == 1


def test_config_aba_between_retries_fails_current_request_then_recovers(tmp_path, monkeypatch):
    from app.jobs import JobCoordinator
    from app.profiles import Profile
    app, provider = setup(tmp_path / 'retry-aba.sqlite')
    original = app.state.coordinator.final_check
    injected = False
    def change_at_final_check(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            changed = replace(app.state.retrieval.settings, return_char_budget=12345)
            other = JobCoordinator(app.state.store, Profile.build(changed, app.state.ingest.embedding_provider))
            other.activate()
            app.state.coordinator.activate()
        return original(*args, **kwargs)
    monkeypatch.setattr(app.state.coordinator, 'final_check', change_at_final_check)
    with TestClient(app) as client:
        first = client.post('/search', json={'user_id': 'u', 'query': 'find target', 'top_k': 100})
        assert first.status_code == 503
        assert first.json()['detail'] == 'query_config_generation_changed'
        assert provider.query_calls == 1
        second = client.post('/search', json={'user_id': 'u', 'query': 'find target', 'top_k': 100})
    assert second.status_code == 200 and second.json()['data']
    assert provider.query_calls == 2
