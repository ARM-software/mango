from collections.abc import Mapping, Iterable
import warnings

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import ParameterGrid


class BatchParameterSampler:

    def __init__(self, param_distribution, n_iter, *, random_state=None):
        if not isinstance(param_distribution, (Mapping)):
            raise TypeError('Parameter distribution is not a dict or '
                            '({!r})'.format(param_distribution))

        dist = param_distribution
        for key in dist:
            if (not isinstance(dist[key], Iterable)
                    and not hasattr(dist[key], 'rvs')):
                raise TypeError('Parameter value is not iterable '
                                'or distribution (key={!r}, value={!r})'
                                .format(key, dist[key]))
        self.n_iter = n_iter
        self.random_state = random_state
        self.dist = param_distribution

    def _is_all_lists(self):
        return all(not hasattr(v, "rvs") for v in self.dist.values())

    def __iter__(self):
        rng = check_random_state(self.random_state)

        # if all distributions are given as lists, we want to sample without
        # replacement
        if self._is_all_lists():
            # look up sampled parameter settings in parameter grid
            param_grid = ParameterGrid(self.dist)
            grid_size = len(param_grid)
            n_iter = self.n_iter

            if grid_size < n_iter:
                warnings.warn(
                    'The total space of parameters %d is smaller '
                    'than n_iter=%d. Running %d iterations. For exhaustive '
                    'searches, use GridSearchCV.'
                    % (grid_size, self.n_iter, grid_size), UserWarning)
                n_iter = grid_size
            for i in sample_without_replacement(grid_size, n_iter,
                                                random_state=rng):
                yield param_grid[i]

        else:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(self.dist.items())
            samples = []
            keys = []
            for k, v in items:
                keys.append(k)
                if hasattr(v, "rvs"):
                    samples.append(v.rvs(random_state=rng, size=self.n_iter))
                else:
                    # forcing object dtype to avoid conversion of mixed datatypes
                    v = np.array(v, dtype=object)
                    samples.append(rng.choice(v, size=self.n_iter))

            for sample in zip(*samples):
                yield dict(zip(keys, sample))

    def __len__(self):
        """Number of points that will be sampled."""
        if self._is_all_lists():
            grid_size = len(ParameterGrid(self.dist))
            return min(self.n_iter, grid_size)
        else:
            return self.n_iter

