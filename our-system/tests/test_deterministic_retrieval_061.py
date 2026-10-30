"""Deterministic retrieval regressions; synthetic records, no real model I/O."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.evidence_budget import evidence_budget, pack_units, InvalidSearchInput
from app.embeddings import EmbeddingConfig, HttpEmbeddingProvider
from app.models import EvidenceCandidate, SearchRequest
from app.profiles import Profile
from app.retrieval import pack_evidence


def settings(path, **kwargs):
    return Settings(db_path=path, profile_id='cycle2_raw', retrieval_policy='evidence_units_v1',
                    return_char_budget=1000000, return_token_budget=65536, packing_policy='skip', **kwargs)


def add(client, texts, user='u', request='r'):
    return client.post('/add', json={'user_id': user, 'request_id': request, 'session_id': 's',
        'messages': [{'role': 'user', 'content': text, 'timestamp': 1704067200000} for text in texts]})


def search(client, query, user='u', k=100):
    return client.post('/search', json={'user_id': user, 'query': query, 'top_k': k})


def hit(key, text, source=None):
    return EvidenceCandidate(id=key, user_id='u', content=text, source_raw_id=source or key,
                             score=1., created_at=1, deletion_state='active', message_index=0)


def test_legacy_identity_and_new_query_only_profile_identity(monkeypatch, tmp_path):
    public = Path(__file__).parents[1] / 'deploy/server/env.server.example'
    values = dict(line.split('=', 1) for line in public.read_text().splitlines()
                  if line and not line.startswith('#') and '=' in line)
    with patch.dict('os.environ', values, clear=True):
        old = Settings.from_env()
    provider = SimpleNamespace(config=old.embedding, implementation_kind='dashscope_http_v1')
    before = Profile.build(old, provider)
    assert before.config_hash == '2fb5b053128ccac29cd9e0cf2b2637020facad8438d546f17e3d76cd768fdddd'
    new = Profile.build(replace(old, retrieval_policy='evidence_units_v1'), provider)
    assert before.config_hash != new.config_hash
    for key in ('model', 'fragment_policy', 'index_version', 'required_routes'):
        assert before.identity[key] == new.identity[key]
    monkeypatch.setenv('AML_RETRIEVAL_POLICY', 'evidence_units_v1')
    assert Settings.from_env().retrieval_policy == 'evidence_units_v1'
    with pytest.raises(ValueError, match='unknown retrieval policy'):
        replace(settings(tmp_path/'a'), retrieval_policy='typo').ensure_phase_a_defaults()


def test_formal_fts_long_query_tail_short_fact_and_isolation(tmp_path):
    path = tmp_path/'memory.sqlite'
    old = create_app(replace(settings(path), retrieval_policy='legacy'))
    query = ' '.join(f'contextword{i}' for i in range(140)) + ' zephyrcobalt'
    with TestClient(old) as client:
        assert add(client, ['zephyrcobalt calibration is 17']).status_code == 200
        assert search(client, query).json()['data'] == []
        old_short = search(client, 'zephyrcobalt').json()
    new = create_app(settings(path))
    new.state.coordinator.certify_routes('u')
    with TestClient(new) as client:
        assert search(client, 'zephyrcobalt').json() == old_short
        assert search(client, query).json()['data'][0]['content'] == 'zephyrcobalt calibration is 17'
        assert search(client, query, user='other').json()['data'] == []
        assert add(client, ['请忘记“zephyrcobalt calibration is 17”'], request='forget').status_code == 200
        assert search(client, 'zephyrcobalt').json()['data'] == []


def test_budget_removes_old_32000_character_bottleneck_on_real_add(tmp_path):
    path = tmp_path/'memory.sqlite'
    old = create_app(replace(settings(path), retrieval_policy='legacy', packing_policy='prefix',
                             return_char_budget=32000, return_token_budget=117760))
    texts = [f'orchid evidence {i}: ' + 'reference measurement ' * 60 for i in range(35)]
    with TestClient(old) as client:
        assert add(client, texts).status_code == 200
        old_hits = search(client, 'orchid').json()['data']
    new = create_app(settings(path))
    new.state.coordinator.certify_routes('u')
    response, trace = new.state.retrieval.search_with_trace(SearchRequest(user_id='u', query='orchid', top_k=100))
    assert len(response.data) > len(old_hits)
    assert len(response.data) == 35
    assert trace['packing']['packed_record_tokens'] <= 65536
    assert trace['output_characters'] > 32000
    assert not new.state.retrieval.settings.enable_extractor


def test_oversize_first_evidence_does_not_suppress_smaller_independent_fact():
    candidates = [hit('oversize', 'x' * 33000), hit('fact', 'The answer is cobalt.')]
    old, _, _ = pack_evidence(candidates, 100, 32000)
    new, _, trace = pack_units(candidates, [], 100, 32000, 65536)
    assert old == [] and [h.id for h in new] == ['fact']
    assert trace['omitted_units'] == [{'seed_id': 'oversize', 'reason': 'unit_over_budget'}]


def test_shared_schema_does_not_merge_all_rows_into_one_huge_component():
    header = hit('schema', 'sample | kg | outcome', 'table')
    rows = [hit(f'row-{i}', f'{i} | 5 | blue', 'table') for i in range(30)]
    groups = [{'seed_id': row.id, 'member_ids': [row.id, header.id]} for row in rows]
    kept, _, _ = pack_units(rows + [header], groups, 5, 32000, 65536)
    assert len(kept) == 5 and len({h.id for h in kept}) == 5
    assert 'schema' in {h.id for h in kept}


def test_filtered_dependencies_and_unrelated_budget_failures_stay_separate():
    candidates = [hit('row', '4 | 9'), hit('independent', 'Tea is preferred.')]
    kept, _, trace = pack_units(candidates,
        [{'seed_id': 'row', 'member_ids': ['row', 'hidden-header']}], 100, 32000, 65536)
    assert [h.id for h in kept] == ['independent']
    assert trace['omitted_units'][0]['reason'] == 'missing_dependency'


def test_summary_spreads_only_existing_sources_and_factual_rank_stays():
    pool = [hit(f'a-{i}', 'Alpha source fact', 'a') for i in range(20)]
    pool += [hit('b', 'Beta source fact'), hit('c', 'Gamma source fact')]
    ordinary, _, _ = pack_units(pool, [], 4, 32000, 65536, query='Alpha fact?')
    summary, _, trace = pack_units(pool, [], 4, 32000, 65536, query='Summarize the entire history.')
    assert [h.id for h in ordinary] == ['a-0', 'a-1', 'a-2', 'a-3']
    assert [h.id for h in summary] == ['a-0', 'a-1', 'b', 'c']
    assert trace['source_spread'] is True


def test_question_options_and_rendered_record_overhead_are_budgeted():
    budget, trace = evidence_budget('context ' * 60000, ['option ' * 1000], 65536)
    assert budget < 65536
    assert budget + trace['query_options_tokens'] + trace['instruction_reserve'] <= 117760
    candidate = hit('long-id-' * 100, 'x')
    kept, _, _ = pack_units([candidate], [], 100, 32000, 8)
    assert kept == []  # one content token alone does not make the record fit
    with pytest.raises(InvalidSearchInput, match='query_context_budget_exceeded'):
        evidence_budget('token ' * 120000, [], 65536)


def test_input_context_exhaustion_is_422_not_empty_success(tmp_path):
    with TestClient(create_app(settings(tmp_path/'memory.sqlite'))) as client:
        result = search(client, 'token ' * 120000)
        assert result.status_code == 422
        assert result.json() == {'detail': 'query_context_budget_exceeded'}
        assert 'Retry-After' not in result.headers


def test_real_native_adapter_mock_long_query_and_permanent_vs_transient_error(monkeypatch, tmp_path):
    monkeypatch.setenv('AML_EMBEDDING_API_KEY', 'synthetic-061-key')
    wire = []; failure = []
    def respond(request):
        body = json.loads(request.content); wire.append(body)
        if failure:
            return httpx.Response(503, json={'error': 'synthetic dependency failure'})
        return httpx.Response(200, json={'output': {'embeddings': [
            {'text_index': i, 'embedding': [1.] + [0.] * 63}
            for i in range(len(body['input']['texts']))]}, 'usage': {'total_tokens': 3}})
    cfg = EmbeddingConfig(provider='dashscope',
        endpoint='https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding', dimensions=64)
    provider = HttpEmbeddingProvider(cfg, transport=httpx.MockTransport(respond))
    candidate = replace(settings(tmp_path/'memory.sqlite'), profile_id='cycle2_dense',
                        retrieval_mode='raw_hybrid_rrf', embedding=cfg)
    app = create_app(candidate, embedding_provider=provider)
    with TestClient(app) as client:
        assert add(client, ['cobalt source']).status_code == 200
        wire.clear()
        result = search(client, 'context ' * 1200 + 'cobalt')
        assert result.status_code == 200
        assert len(wire) == 1 and len(wire[0]['input']['texts']) == 2
        wire.clear()
        result = search(client, 'context ' * 11000)
        assert result.status_code == 422
        assert result.json() == {'detail': 'query_window_budget_exceeded'}
        assert wire == [] and 'Retry-After' not in result.headers
        failure.append(True)
        result = search(client, 'cobalt')
        assert result.status_code == 503 and result.headers['Retry-After'] == '1'


@pytest.mark.parametrize('query', ['When did Orchid start?', 'What is the current Orchid status?'])
def test_source_dates_retain_received_text_and_do_not_invent_event_time(tmp_path, query):
    app = create_app(settings(tmp_path/'m.sqlite', temporal_source_dates=True))
    with TestClient(app) as client:
        assert add(client, ['Orchid started last month.']).status_code == 200
        result = search(client, query)
        assert result.status_code == 200
        assert result.json()['data'][0]['content'] == '[Source recorded: 2024-01-01 UTC]\nOrchid started last month.'


def test_table_and_rule_completion_through_real_search_and_top_one(tmp_path):
    table = ('Units: kg\n| Product | Quantity |\n| --- | --- |\n'
             + ''.join(f'| item{i:03d} | {i:03d} |\n' for i in range(105))
             + '| tailneedle | 4321 |\n')
    rules = ('# Dispatch rules\nPrerequisite: the operator must be certified.\n'
             + 'Routine preparation notes are recorded before dispatch.\n' * 27
             + 'The operator must run rulemarker before departure.\n'
             + 'Routine review notes are retained for inspection.\n' * 27
             + 'Exception: unless the alarm is clear, departure is prohibited.\n')
    app = create_app(settings(tmp_path/'m.sqlite'))
    with TestClient(app) as client:
        assert add(client, [table, rules]).status_code == 200
        response = search(client, 'tailneedle')
        assert response.status_code == 200
        data = response.json()['data']
        assert any('tailneedle' in h['content'] for h in data)
        assert any('Units: kg' in h['content'] and '| Product |' in h['content'] for h in data)
        # A one-hit caller cannot receive a two-part schema unit. Preserve its
        # independent recalled seed; do not turn it into an empty response.
        top_one = search(client, 'tailneedle', k=1)
        assert len(top_one.json()['data']) == 1
        assert 'tailneedle' in top_one.json()['data'][0]['content']
        data = search(client, 'rulemarker').json()['data']
        assert any('Prerequisite:' in h['content'] for h in data)
        assert any('unless the alarm' in h['content'] for h in data)


def test_final_fence_retries_after_delete_during_structural_expansion(tmp_path, monkeypatch):
    app = create_app(settings(tmp_path/'m.sqlite'))
    with TestClient(app) as client:
        assert add(client, ['needle revoked source.']).status_code == 200
        expand = app.state.store.expand_evidence_units
        deleted = []
        def race(*args, **kwargs):
            result = expand(*args, **kwargs)
            if not deleted:
                deleted.append(True)
                assert add(client, ['请忘记“needle revoked source.”'], request='delete').status_code == 200
            return result
        monkeypatch.setattr(app.state.store, 'expand_evidence_units', race)
        response = search(client, 'needle')
        assert response.status_code == 200 and response.json()['data'] == []
        assert deleted == [True]


@pytest.mark.parametrize('mechanism', ['parent_evidence', 'sentence_index', 'context_neighbor_radius'])
def test_existing_legal_raw_routes_survive_structural_policy(tmp_path, mechanism):
    results = []
    for policy in ('legacy', 'evidence_units_v1'):
        kwargs = {mechanism: 1 if mechanism == 'context_neighbor_radius' else True}
        cfg = replace(settings(tmp_path/(policy+'.sqlite'), **kwargs), retrieval_policy=policy)
        with TestClient(create_app(cfg)) as client:
            assert add(client, ['Orchid calibration is 17.', 'The calibration exception is holiday.']).status_code == 200
            response = search(client, 'Orchid')
            assert response.status_code == 200 and response.json()['data']
            results.append({h['content'] for h in response.json()['data']})
    assert results[0] == results[1]


def test_raw_retention_validates_provenance_permissions_and_deletion(tmp_path):
    from contextlib import closing
    from app.evidence_navigation import raw_candidate
    from app.evidence_units import expand_units
    app = create_app(settings(tmp_path/'m.sqlite', parent_evidence=True))
    with TestClient(app) as client:
        assert add(client, ['Orchid calibration is 17.']).status_code == 200
        with closing(app.state.store._connect()) as db:
            row = db.execute('SELECT m.*,p.source_version,p.content_sha256 FROM raw_messages m '
                             'JOIN source_policies p ON m.id=p.raw_id AND m.user_id=p.user_id').fetchone()
            raw = raw_candidate(row)
            assert expand_units(db, 'u', [raw])[0] == [raw]
            for changes in ({'user_id':'other'}, {'content':raw.content+' forged'},
                            {'source_version':raw.source_version+1}, {'source_sha256':'x'*64},
                            {'role':'assistant'}, {'source_timestamp':0}):
                assert expand_units(db, 'u', [raw.model_copy(update=changes)])[0] == []
            db.execute("UPDATE source_policies SET disclosure_policy='blocked'")
            assert expand_units(db, 'u', [raw])[0] == []
            db.execute("UPDATE source_policies SET disclosure_policy='within_user',retraction_state='retracted'")
            assert expand_units(db, 'u', [raw])[0] == []
            db.execute("UPDATE source_policies SET retraction_state='not_retracted'")
        assert add(client, ['请忘记“Orchid calibration is 17.”'], request='erase').status_code == 200
        with closing(app.state.store._connect()) as db:
            assert expand_units(db, 'u', [raw])[0] == []


@pytest.mark.parametrize('dates', [False, True])
def test_record_overhead_prevents_optional_group_from_swallowing_independent_seed(tmp_path, dates):
    table = ('Units: kg\n| Product | Quantity |\n| --- | --- |\n'
        + ''.join(f'| item{i:03d} | {i:03d} |\n' for i in range(105)) + '| tailneedle | 4321 |\n')
    cfg = replace(settings(tmp_path/'m.sqlite', temporal_source_dates=dates), return_token_budget=819)
    app = create_app(cfg)
    with TestClient(app) as client:
        assert add(client, [table]).status_code == 200
        response, trace = app.state.retrieval.search_with_trace(
            SearchRequest(user_id='u', query='When tailneedle', top_k=100))
        assert any('tailneedle' in h.content for h in response.data)
        assert trace['packing']['packed_record_tokens'] <= 819
        assert trace['evidence_units']['hard_groups'] == 0


def test_packing_does_not_rescan_all_candidates_per_hit():
    class Counted(list):
        visits = 0
        def __iter__(self):
            for item in super().__iter__():
                self.visits += 1
                yield item
    pool = Counted(hit(str(i), 'short fact') for i in range(200))
    kept, _, _ = pack_units(pool, [], 1, 32000, 65536)
    assert len(kept) == 1
    assert pool.visits <= 3*len(pool)
