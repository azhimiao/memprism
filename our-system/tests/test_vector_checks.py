from app.vector_checks import valid_vector_json,_cache


def test_cache_revalidates_full_json_mutation_and_dimensions_without_vector_retention():
    assert valid_vector_json('[1.0,0.0]',2)
    assert not valid_vector_json('[0.0,0.0]',2)
    assert not valid_vector_json('[1.0,0.0]',3)
    assert not valid_vector_json('[true,0.0]',2)
    assert not valid_vector_json('[1e400,0.0]',2)
    assert all(isinstance(key[1],str) and len(key[1])==64 and isinstance(value,bool) for key,value in _cache.items())
