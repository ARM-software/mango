import functools


def serial(func):
    func.is_wrapped = True

    @functools.wraps(func)
    def wrapper(params_batch):
        return [func(**params) for params in params_batch]

    return wrapper


def parallel(n_jobs):
    from joblib import Parallel, delayed, cpu_count

    assert n_jobs != 0
    assert n_jobs >= -1

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


def celery(n_jobs, timeout=None):
    import celery

    def decorator(func):
        func.is_wrapped = True
        func.batch_size = n_jobs

        @functools.wraps(func)
        def wrapper(params_batch):
            async_results = [func(**params) for params in params_batch]
            results = []
            params_evaluated = []
            for params, async_result in zip(params_batch, async_results):
                try:
                    result = async_result.get(timeout=timeout)
                    params_evaluated.append(params)
                    results.append(result)
                except celery.exceptions.TimeoutError:
                    # ignore the timed out jobs as we are returning (params, result) tuple
                    continue

            return params_evaluated, results

        return wrapper

    return decorator


def custom(n_jobs):
    assert n_jobs > 0

    def decorator(func):
        func.is_wrapped = True
        func.batch_size = n_jobs

        @functools.wraps(func)
        def wrapper(params_batch):
            return func(params_batch)

        return wrapper

    return decorator

