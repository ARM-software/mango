import functools


def simple_local(func):
    func.is_wrapped = True

    @functools.wraps(func)
    def wrapper(params_batch):
        return [func(**params) for params in params_batch]

    return wrapper


def parallel_local(n_jobs):
    from joblib import Parallel, delayed, cpu_count

    def decorator(func):
        func.is_wrapped = True
        if n_jobs == -1:
            func.batch_size = cpu_count()
        else:
            func.batch_size = n_jobs

        @functools.wraps(func)
        def wrapper(params_batch):
            return Parallel(n_jobs)(delayed(func)(**params) for params in params_batch)

        return wrapper

    return decorator
