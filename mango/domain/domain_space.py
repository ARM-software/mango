from collections.abc import Mapping, Iterable
import math
import numpy as np
from collections.abc import Callable
import warnings
from itertools import compress
from functools import cached_property

from scipy.stats._distn_infrastructure import rv_frozen

from mango.domain.parameter_sampler import parameter_sampler


class DomainSpace:
    def __init__(
        self,
        param_dist: dict,
        *,
        param_sampler: Callable = parameter_sampler,
        constraint: Callable = None,
        constraint_max_retries: int = 10,
    ):
        self.dist = param_dist
        if not isinstance(self.dist, (Mapping)):
            raise TypeError(
                "Parameter distribution is not a dict or " "({!r})".format(self.dist)
            )

        for key in self.dist:
            if not isinstance(self.dist[key], Iterable) and not hasattr(
                self.dist[key], "rvs"
            ):
                raise TypeError(
                    "Parameter value is not iterable "
                    "or distribution (key={!r}, value={!r})".format(key, self.dist[key])
                )

        self.param_sampler = param_sampler
        self.constraint = constraint
        self.constraint_max_tries = constraint_max_retries
        self.domain_size = self.calc_domain_size(self.dist)
        self.categorical_params, self.integer_params = self.classify_parameters(
            self.dist
        )

    @cached_property
    def categorical_index_lookup(self):
        res = {}
        for param in self.categorical_params:
            res[param] = {ele: pos for pos, ele in enumerate(self.dist[param])}
        return res

    def get_domain(self):
        return self.get_random_sample(self.domain_size)

    def get_random_sample(self, size):
        if self.constraint is None:
            return self._get_random_sample(size)

        samples = []
        n_tries = 0
        factor = 2.0
        while len(samples) < size and n_tries < self.constraint_max_tries:
            n_samples = math.ceil((size - len(samples)) * factor)
            raw_samples = self._get_random_sample(n_samples)
            _filters = self.constraint(raw_samples)
            filtered_samples = list(compress(raw_samples, _filters))
            if len(filtered_samples) == 0:
                factor *= 10
            else:
                factor = max(len(raw_samples) / len(filtered_samples), 1.0)
            samples += filtered_samples
            n_tries += 1
        if len(samples) < size:
            warnings.warn(
                f"Could not get {size} samples that satisfy the constraint"
                f"even after {n_tries * size} random samples ",
                UserWarning,
            )
        return samples[:size]

    def _get_random_sample(self, size):
        return self.param_sampler(self.dist, size)

    def convert_GP_space(self, domain_list):
        """
        convert the hyperparameters from the param_dict space to the GP space, by converting the
        categorical variables to one hotencoded, and return a numpy array which can be used to train the GP

        input is the domain_list generated using the Parameter Sampler.
        """
        categorical_params = self.categorical_params

        X = []
        for domain in domain_list:
            curr_x = []
            # for x in domain:
            for x in sorted(domain.keys()):
                # this value can be directly used, for int too, we will consider it as a float for GP

                if x not in categorical_params:
                    curr_x.append(domain[x])

                # this is a categorical variable which require special handling
                elif x in categorical_params:
                    size = len(self.dist[x])  # total number of categories.
                    # we need to see the index where domain[x] appears in param_dict[x]
                    index = self.categorical_index_lookup[x][domain[x]]
                    listofzeros = [0.0] * size
                    # We will set the value to one for one hot encoding
                    listofzeros[index] = 1.0
                    # expanding current list
                    curr_x = curr_x + listofzeros

            X.append(curr_x)

        X = np.array(X)
        return X

    def convert_PS_space(self, X_gp):
        """
        Convert from the X_gp space which is a numpy array that can be given input to the gaussian process
        to the parameter sampler space which is a list of the dict

        We have to reverse the one-hotencoded transformation of the categories to the category name
        """
        X_ps = []

        categorical_params = self.categorical_params
        int_params = self.integer_params
        param_dict = self.dist

        for i in range(X_gp.shape[0]):

            curr_x_gp = X_gp[i]
            # we will create a list of dict, same as parameter samplers
            curr_x_ps = dict()
            index = 0

            # every sample is from the param_dict
            for par in sorted(param_dict.keys()):
                # this has to have integer values
                if par in int_params:
                    curr_x_ps[par] = int(round(curr_x_gp[index]))
                    index = index + 1

                # this par is a categorical variable and we need to handle it carefully
                elif par in categorical_params:
                    size = len(param_dict[par])  # total number of categories.
                    one_hot_encoded = curr_x_gp[index : index + size]
                    category_idx = np.argmax(one_hot_encoded)
                    category_val = param_dict[par][category_idx]
                    curr_x_ps[par] = category_val
                    # we have processed the entire one-hotencoded
                    index = index + size
                # this is a float value
                else:
                    curr_x_ps[par] = curr_x_gp[index]
                    index = index + 1

            X_ps.append(curr_x_ps)
        return X_ps

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
