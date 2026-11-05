"""Attack stale proofs, mutation, dimension confusion, retention and telemetry."""
import json
import asyncio
import pytest
from app import vector_checks as vc, diagnostics as d
from app.embeddings import EmbeddingFailure
from app.exact_dense import exact_top_scores

def test_warm_proof_skips_components_but_not_fresh_decode(monkeypatch):
    text='[0.123456789123,0.456789123456]'
    assert vc.valid_vector_json(text,2)
    monkeypatch.setattr(vc,'parse_vector_json',lambda _:pytest.fail('already verified exact hash'))
    a=vc.decode_validated_vector(text,2);b=vc.decode_validated_vector(text,2)
    a[0]=999
    assert b==json.loads(text)
    assert a is not b
    assert all(type(k[1]) is str and len(k[1])==64 and type(v) is bool for k,v in vc._cache.items())

@pytest.mark.parametrize('changed',['[true,1]','["0.123456789123",1]','[NaN,1]','[0,0]','[1]'])
def test_valid_proof_cannot_authorize_mutated_vector(changed):
    assert vc.valid_vector_json('[0.123456789123,1]',2)
    assert not vc.valid_vector_json(changed,2)
    with pytest.raises(EmbeddingFailure):
        exact_top_scores([changed],[1,0],2,1,[(0,)])

def test_dimension_is_part_of_proof_and_error_semantics():
    text='[0.345678912345,1]'
    assert vc.valid_vector_json(text,2)
    assert not vc.valid_vector_json(text,3)
    with pytest.raises(EmbeddingFailure) as exc:
        exact_top_scores([text],[1,0,0],3,1,[(0,)])
    assert exc.value.code=='dimension_mismatch'

def test_eviction_remains_bounded_and_forces_fresh_validation(monkeypatch):
    monkeypatch.setattr(vc,'_CACHE_LIMIT',2)
    for i in range(5):assert vc.valid_vector_json(json.dumps([1,i+.234567891]),2)
    assert len(vc._cache)<=2 and vc._cache_bytes<=vc._CACHE_MAX_BYTES
    assert not vc.valid_vector_json('[true,0]',2)

def test_safe_stage_and_builtin_error_metadata_never_include_message(capsys):
    token=d.begin('/search')
    try:
        d.stage('search_snapshot')
        try:raise SystemError('PRIVATE-SECRET-AND-TEXT')
        except SystemError as exc:d.failure(exc,'search_failed')
        d.emit('input_shape',returned_items=10,packed_record_tokens=123,output_characters=456,secret='PRIVATE-SECRET')
        d.flush_stage()
    finally:d.end(token)
    out=capsys.readouterr().err
    assert 'PRIVATE-SECRET' not in out
    rows=[json.loads(x.split('AML_DIAG ',1)[1]) for x in out.splitlines()]
    assert any(x.get('exception_type')=='SystemError' for x in rows)
    assert any(x['event']=='stage_timing' and x['elapsed_ms']>=0 for x in rows)
    assert any(x.get('packed_record_tokens')==123 for x in rows)

def test_transport_counts_complete_wire_response_without_reading_content(capsys):
    from app.transport import TransportGuard
    from app.config import Settings
    async def app(scope,receive,send):
        await receive()
        await send(dict(type='http.response.start',status=200,headers=[]))
        await send(dict(type='http.response.body',body=b'abc',more_body=True))
        await send(dict(type='http.response.body',body=b'defg',more_body=False))
    async def check():
        async def receive():return dict(type='http.request',body=b'{}')
        async def send(event):pass
        await TransportGuard(app,Settings(db_path=':memory:',auth_mode='none'))(
            dict(type='http',path='/search',headers=[]),receive,send)
    asyncio.run(check())
    rows=[json.loads(x.split('AML_DIAG ',1)[1]) for x in capsys.readouterr().err.splitlines()]
    last=next(x for x in rows if x['event']=='request_finished')
    assert last['response_bytes']==7 and last['response_complete'] is True
