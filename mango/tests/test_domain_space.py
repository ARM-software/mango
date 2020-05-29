import numpy as np
from scipy.stats import uniform, loguniform

from mango.domain.domain_space import domain_space


def test_domain():
    params = {
        'a': range(10),
        'b': np.random.randint(0, 1e6, size=50),
        'd': [1, 0.1, 2.0],
        'e': ['a', 1, 'b', 'c', None],
        'f': ['1', '-1'],
        'h': [True, False]
    }
    ds = domain_space(params, domain_size=1000)
    assert all(k in ds.mapping_int for k in ['a', 'b'])
    assert all(k in ds.mapping_categorical for k in ['d', 'e', 'f', 'h'])
    samples = ds.get_domain()

    for sample in samples:
        for param in params.keys():
            assert (sample[param] in params[param])

    params = {
        'a': [1],
    }
    ds = domain_space(params, domain_size=1000)
    assert all(k in ds.mapping_int for k in ['a'])
    samples = ds.get_domain()

    for sample in samples:
        for param in params.keys():
            assert (sample[param] in params[param])


def test_gp_samples_to_params():
    space = {
        'a': range(10),
        'b': uniform(-10, 20),
        'c': ['cat1', 1, 'cat2'],
        'e': [1, 2, 3],
        'f': ['const'],
        'g': loguniform(0.001, 100),
        'h': [10]
    }
    X = np.array([
        # 4, -8, 'cat2', 1, 'const', 1 , 10
        [0.4444, 0.1, 0, 0, 1, 0, 1, 0.6, 0],
        # 0, -10.0, 'cat1', 3, 'const', 0.001 , 10
        [0.0, 0.0, 1, 0, 0, 1, 1, 0.0, 0],
        # 9, 10.0, 1, 2, 'const', 100 , 10
        [1.0, 1.0, 0, 1, 0, 0.5, 1, 1.0, 0],
    ])

    expected = [
        dict(a=4, b=-8.0, c='cat2', e=1, f='const', g=1, h=10),
        dict(a=0, b=-10.0, c='cat1', e=3, f='const', g=.001, h=10),
        dict(a=9, b=10.0, c=1, e=2, f='const', g=100, h=10),
    ]

    ds = domain_space(space, domain_size=1000)

    params = ds.convert_to_params(X)

    for act, exp in zip(params, expected):
        for k, v in act.items():
            if k == 'g':
                assert np.isclose(v, exp[k])
            else:
                assert v == exp[k]


def test_gp_space():
    space = {
        'f': range(10),
        'h': uniform(-10, 20),
        'e': ['cat1', 1, 'cat2'],
        'c': [1, 2, 3],
        'a': ['const'],
        'g': loguniform(0.001, 100),
        'b': [10],
        'd': uniform(0, 1),
        'i': [True, False]
    }

    ds = domain_space(space, domain_size=10000)
    X = ds.sample_gp_space()

    assert (X <= 1.0).all()
    assert (X >= 0.0).all()
    assert (X[:, 0] == 1.).all()  # a
    assert (X[:, 1] == 0.).all()  # b
    assert np.isin(X[:, 2], [0.0, 0.5, 1.0]).all() # c
    assert np.isin(X[:, 4:7], np.eye(3)).all()  # e
    assert X.shape == (ds.domain_size, 12)

    params = ds.convert_to_params(X)

    for param in params:
        assert param['a'] == 'const'
        assert param['b'] == 10
        assert param['c'] in space['c']
        assert 0.0 <= param['d'] <= 1.0
        assert param['e'] in space['e']
        assert param['f'] in space['f']
        assert 0.001 <= param['g'] <= 100
        assert -10 <= param['h'] <= 10
        assert param['i'] in space['i']

    X2 = ds.convert_to_gp(params)
    assert np.isclose(X2, X).all()


