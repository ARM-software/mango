from abc import ABCMeta, abstractmethod

import warnings

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import ParameterGrid


def parameter_sampler(
    param_dist: dict, n_size: int, *, random_state=None
) -> list[dict]:
    """
    :param param_dist: dict of parameter distributions
    :param n_size: number of samples to return
    :param random_state: specify to reproduce results
    :return: list of samples mapping each parameter to one of its
        allowed values.
    """
    rng = check_random_state(random_state)

    # if all distributions are given as lists, we want to sample without
    # replacement
    if all(not hasattr(v, "rvs") for v in param_dist.values()):
        # look up sampled parameter settings in parameter grid
        param_grid = ParameterGrid(param_dist)
        grid_size = len(param_grid)

        if grid_size < n_size:
            warnings.warn(
                "The total space of parameters %d is smaller "
                "than n_iter=%d. Running %d iterations. For exhaustive "
                "searches, use GridSearchCV." % (grid_size, n_size, grid_size),
                UserWarning,
            )
            n_size = grid_size
        return [
            param_grid[i]
            for i in sample_without_replacement(grid_size, n_size, random_state=rng)
        ]

    else:
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(param_dist.items())
        samples = []
        keys = []
        for k, v in items:
            keys.append(k)
            if hasattr(v, "rvs"):
                samples.append(v.rvs(random_state=rng, size=n_size).tolist())
            else:
                # forcing object dtype to avoid conversion of mixed datatypes
                v = np.array(v, dtype=object)
                samples.append(rng.choice(v, size=n_size))

        return [dict(zip(keys, sample)) for sample in zip(*samples)]
