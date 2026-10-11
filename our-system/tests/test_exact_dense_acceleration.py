import json
import math
import random
import pytest
from app.embeddings import normalize_vector, EmbeddingFailure
from app.exact_dense import exact_top_scores


def reference(vectors, query, dimensions, limit, keys):
    q = normalize_vector(query, dimensions)
    out = [(i, max(-1., min(1., math.fsum(a*b for a,b in zip(q,normalize_vector(v, dimensions))))))
           for i,v in enumerate(vectors)]
    return sorted(out, key=lambda x:(-x[1],*keys[x[0]]))[:limit]


@pytest.mark.parametrize('dimensions', [2, 17, 4096])
def test_exact_scores_same_order_with_ties_near_boundary_and_extremes(dimensions):
    rng = random.Random(38+dimensions)
    vectors = [[rng.uniform(-1,1) for _ in range(dimensions)] for _ in range(105)]
    vectors.extend([vectors[0], [v*1e300 for v in vectors[1]], [v*1e-300 for v in vectors[2]]])
    query = [rng.uniform(-1,1) for _ in range(dimensions)]
    keys = [(len(vectors)-i,str(i)) for i in range(len(vectors))]
    for limit in (1,20,100,len(vectors)):
        assert exact_top_scores([json.dumps(v) for v in vectors],query,dimensions,limit,keys) == reference(vectors,query,dimensions,limit,keys)


@pytest.mark.parametrize('bad', ['[true,0]', '[0,0]', '[1e400,0]', '["1",0]', '[1]', '[[1],0]', '[NaN,1]', '{"a":1}'])
def test_invalid_losing_vector_also_fails(bad):
    with pytest.raises(EmbeddingFailure):
        exact_top_scores(['[1,0]',bad],[1,0],2,1,[(0,),(1,)])


def test_exact_ties_cannot_be_lost_by_screening():
    vectors = ['[1,1]']*250
    keys = [(249-i,) for i in range(250)]
    out = exact_top_scores(vectors,[1,1],2,20,keys)
    assert [i for i,_ in out] == list(range(249,229,-1))


def test_every_invocation_checks_mutation_without_retaining_vectors():
    assert exact_top_scores(['[1,0]'],[1,0],2,1,[(0,)]) == [(0,1.)]
    assert exact_top_scores(['[0,1]'],[1,0],2,1,[(0,)]) == [(0,0.)]
    with pytest.raises(EmbeddingFailure):
        exact_top_scores(['[0,0]'],[1,0],2,1,[(0,)])
