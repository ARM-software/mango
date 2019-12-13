import functools


def simple_local(func):
    func.is_wrapped = True

    @functools.wraps(func)
    def wrapper(params_batch):
        return [func(**params) for params in params_batch]

    return wrapper
