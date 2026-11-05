"""Numerical and adversarial checks on synthetic vectors; no production DB IO."""
from concurrent.futures import ThreadPoolExecutor
import json,math,random
import pytest
from app.embeddings import EmbeddingFailure,normalize_vector
from app.exact_dense import exact_top_scores
from app.vector_checks import valid_vector_json
from app.vector_json import parse_vector_json


@pytest.mark.parametrize('text',['[true,0]','[false,1]','["1",0]','null','{}','[[]]',
    '[NaN,0]','[Infinity,0]','[-Infinity,0]','[1e9999,0]','[1,]','[+1,0]',
    '[1 2]','[1,0]garbage','['+('9'*400)+',0]'])
def test_rejects_invalid_without_native_schema(text):
    with pytest.raises((ValueError,TypeError,OverflowError,RecursionError)):
        parse_vector_json(text)
    assert valid_vector_json(text,2) is False
    with pytest.raises(EmbeddingFailure):
        exact_top_scores(['[1,0]',text],[1,0],2,1,[(0,),(1,)])


def test_deep_json_is_rejected_at_vector_boundary():
    text='['*1500+'0'+']'*1500
    assert not valid_vector_json(text,2)
    with pytest.raises(EmbeddingFailure):
        exact_top_scores([text],[1,0],2,1,[(0,)])


def test_strict_parser_retains_float_domain_and_subnormals():
    raw='[1,-2,0.1,-0.0,5e-324,1e308,-1e-300]'
    actual=parse_vector_json(raw)
    assert all(type(v) is float for v in actual)
    assert [v.hex() for v in actual]==[float(v).hex() for v in json.loads(raw)]


def test_16_workers_repeat_exact_scores_without_shared_native_validator():
    rng=random.Random(101)
    dims=64
    vectors=[[rng.uniform(-1,1) for _ in range(dims)] for i in range(96)]
    vectors.extend([vectors[0],vectors[0],[x*1e300 for x in vectors[1]],[x*1e-300 for x in vectors[2]]])
    wire=[json.dumps(v) for v in vectors];keys=[(len(vectors)-i,str(i)) for i in range(len(vectors))]
    raw_query=[rng.uniform(-1,1) for _ in range(dims)]
    q=normalize_vector(raw_query,dims)
    expected=sorted([(i,max(-1.,min(1.,math.fsum(a*b for a,b in zip(q,normalize_vector(v,dims))))))
        for i,v in enumerate(vectors)],key=lambda r:(-r[1],*keys[r[0]]))[:20]
    expected=[(i,s.hex()) for i,s in expected]
    def check(_):
        assert all(valid_vector_json(v,dims) for v in wire)
        result=exact_top_scores(wire,raw_query,dims,20,keys)
        assert [(i,s.hex()) for i,s in result]==expected
        return len(result)
    with ThreadPoolExecutor(max_workers=16) as pool:
        assert list(pool.map(check,range(128)))==[20]*128
