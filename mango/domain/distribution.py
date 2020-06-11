from scipy.stats import loguniform as _loguniform


def loguniform(a, b):
    return _loguniform(10 ** a, 10 ** (a + b))
