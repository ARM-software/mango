from abc import ABCMeta, abstractmethod
from collections.abc import Mapping, Iterable
import warnings

from scipy.stats._distn_infrastructure import rv_frozen
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import ParameterGrid


class AbstractParameterSampler(metaclass=ABCMeta):
    dist: dict
    categorical_params: set[str]
    integer_params: set[str]
    domain_size: int
    categorical_index_lookup: dict

    @abstractmethod
    def __init__(self, param_distribution: dict, *, random_state=None):
        pass

    @abstractmethod
    def sample(self, n_size: int) -> list:
        pass


class ParameterSampler(AbstractParameterSampler):

    def __init__(self, param_distribution: dict, *, random_state=None):
        if not isinstance(param_distribution, (Mapping)):
            raise TypeError(
                "Parameter distribution is not a dict or "
                "({!r})".format(param_distribution)
            )

        dist = param_distribution
        for key in dist:
            if not isinstance(dist[key], Iterable) and not hasattr(dist[key], "rvs"):
                raise TypeError(
                    "Parameter value is not iterable "
                    "or distribution (key={!r}, value={!r})".format(key, dist[key])
                )
        self.random_state = random_state
        self.dist = param_distribution

        self.domain_size = self.calc_domain_size(self.dist)
        self.categorical_params, self.integer_params = self.classify_parameters(
            self.dist
        )
        # creating index to value map for categorical params for efficiency lookup
        self.categorical_index_lookup = dict()
        for param in self.categorical_params:
            self.categorical_index_lookup[param] = {
                ele: pos for pos, ele in enumerate(self.dist[param])
            }

    def _is_all_lists(self):
        return all(not hasattr(v, "rvs") for v in self.dist.values())

    def sample(self, n_size: int) -> list[dict]:
        """
        :param n_size: number of samples to return
        :return: list of samples mapping each parameter to one of its
            allowed values.
        """
        rng = check_random_state(self.random_state)

        # if all distributions are given as lists, we want to sample without
        # replacement
        if self._is_all_lists():
            # look up sampled parameter settings in parameter grid
            param_grid = ParameterGrid(self.dist)
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
            items = sorted(self.dist.items())
            samples = []
            keys = []
            for k, v in items:
                keys.append(k)
                if hasattr(v, "rvs"):
                    samples.append(v.rvs(random_state=rng, size=n_size))
                else:
                    # forcing object dtype to avoid conversion of mixed datatypes
                    v = np.array(v, dtype=object)
                    samples.append(rng.choice(v, size=n_size))

            return [dict(zip(keys, sample)) for sample in zip(*samples)]

    @staticmethod
    def calc_domain_size(param_dict: dict) -> int:
        """
        Return an estimate of number of points in the search space.
        """
        # Minimum and maximum domain size
        domain_min = 50000
        domain_max = 500000

        ans = domain_min

        for par in param_dict:
            if isinstance(param_dict[par], rv_frozen):
                distrib = param_dict[par]
                loc, scale = distrib.args
                min_scale = 1
                scale = int(scale)
                if scale < min_scale:
                    scale = min_scale

                ans = ans * scale * 50

            elif isinstance(param_dict[par], range):
                ans = ans * len(param_dict[par])

            elif isinstance(param_dict[par], list):
                ans = ans * len(param_dict[par])

        if ans < domain_min:
            ans = domain_min

        if ans > domain_max:
            ans = domain_max

        return ans

    @staticmethod
    def classify_parameters(param_dict: dict) -> (list, list):
        """
        Identify parameters that are categorical or integer types.
        Categorical values/discrete values are considered from the list of each value being str.
        Integer values are considered from list of each value as int or from a range

        :param param_dict: dictionary of parameter distributions
        :return: a tuple of two sets where first element is the set of parameter that are categorical
        and second element is the set of parameters that are integer
        """
        categorical_params = set()
        int_params = set()

        for par in param_dict:
            if isinstance(param_dict[par], rv_frozen):
                # FIXME: what if the distribution generators ints , GP would convert it to float
                pass  # we are not doing anything at present, and will directly use its value for GP.

            elif isinstance(param_dict[par], range):
                int_params.add(par)

            elif isinstance(param_dict[par], Iterable):
                # for list with all int we are considering it as non-categorical
                try:
                    # this check takes care of numpy ints as well
                    all_int = all(
                        x == int(x) and type(x) != bool for x in param_dict[par]
                    )
                except (ValueError, TypeError):
                    all_int = False

                if all_int:
                    int_params.add(par)
                # For lists with mixed type, floats or strings we consider them categorical or discrete
                else:
                    categorical_params.add(par)

        return (categorical_params, int_params)
