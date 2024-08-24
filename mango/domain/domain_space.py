from collections.abc import Mapping, Iterable
import math
import numpy as np
from collections.abc import Callable
import warnings
from itertools import compress
from functools import cached_property
import warnings

from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen
from sklearn.preprocessing import MinMaxScaler

from mango.domain.parameter_sampler import parameter_sampler


class DomainSpace:
    def __init__(
        self,
        param_dist: dict,
        *,
        param_sampler: Callable = parameter_sampler,
        constraint: Callable = None,
        constraint_max_retries: int = 10,
        scale_params: bool = False,
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
        self.scale_params = scale_params
        (
            self.categorical_params,
            self.integer_params,
            self.multivariate_params,
            self.univariate_params,
        ) = self.classify_parameters(self.dist)
        self.domain_size = self.calc_domain_size()

    @cached_property
    def categorical_index_lookup(self):
        res = {}
        for param in self.categorical_params:
            res[param] = {ele: pos for pos, ele in enumerate(self.dist[param])}
        return res

    @cached_property
    def multivariate_param_dimensions(self):
        res = {}
        for param in self.multivariate_params:
            res[param] = len(self.dist[param].mean())
        return res

    @cached_property
    def gp_scaler(self):
        data = self.get_domain()
        x_gp = self._convert_GP_space(data)
        scaler = MinMaxScaler()
        scaler.fit(x_gp)
        return scaler

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

    def _convert_GP_space(self, domain_list):
        categorical_params = self.categorical_params

        X = []
        for domain in domain_list:
            curr_x = []
            for x in sorted(domain.keys()):
                # this is a categorical variable which require special handling
                if x in categorical_params:
                    size = len(self.dist[x])  # total number of categories.
                    # we need to see the index where domain[x] appears in param_dict[x]
                    index = self.categorical_index_lookup[x][domain[x]]
                    listofzeros = [0.0] * size
                    # We will set the value to one for one hot encoding
                    listofzeros[index] = 1.0
                    # expanding current list
                    curr_x = curr_x + listofzeros
                elif x in self.multivariate_params:
                    curr_x = curr_x + list(domain[x])

                # this value can be directly used, for int too, we will consider it as a float for GP
                else:
                    curr_x.append(domain[x])

            X.append(curr_x)

        X = np.array(X)
        return X

    def convert_GP_space(self, domain_list):
        """
        convert the hyperparameters from the param_dict space to the GP space, by converting the
        categorical variables to one hotencoded, and return a numpy array which can be used to train the GP

        input is the domain_list generated using the Parameter Sampler.
        """
        X = self._convert_GP_space(domain_list)
        if self.scale_params:
            X = self.gp_scaler.transform(X)
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
        mv_params = self.multivariate_params
        param_dict = self.dist

        if self.scale_params:
            X_gp = self.gp_scaler.inverse_transform(X_gp)

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

                elif par in mv_params:
                    dim = self.multivariate_param_dimensions[par]
                    curr_x_ps[par] = curr_x_gp[index : index + dim].tolist()
                    index += dim

                # this is a float value
                else:
                    curr_x_ps[par] = curr_x_gp[index]
                    index = index + 1

            X_ps.append(curr_x_ps)
        return X_ps

    def calc_domain_size(self) -> int:
        """
        Return an estimate of number of points in the search space.
        """
        # Minimum and maximum domain size
        domain_min = 50000
        domain_max = 500000

        ans = 1

        for par, dist in self.dist.items():
            if par in self.univariate_params:
                loc, scale = dist.args
                min_scale = 1
                scale = int(scale)
                if scale < min_scale:
                    scale = min_scale

                ans = ans * scale * 50

            elif par in self.multivariate_params:
                # FIXME:  assuming scale 10 for multivariate as don't see a way to get it from the dist
                ans = ans * 10

            elif par in self.integer_params:
                ans = ans * len(dist)

            elif par in self.categorical_params:
                ans = ans * len(dist)

        if ans < domain_min:
            ans = domain_min

        if ans > domain_max:
            ans = domain_max

        return ans

    @staticmethod
    def classify_parameters(param_dict: dict) -> (set, set, set, set):
        """
        Identify parameters that are categorical or integer types.
        Categorical values/discrete values are considered from the list of each value being str
        Integer values are considered from list of each value as int or from a range
        Multivariate values are random variables defined using scipy stats multivariate distribution
        Univariate values are random variable defined using scipy stats distribution

        :param param_dict: dictionary of parameter distributions
        :return: a tuple of four sets where first element is the set of parameter that are categorical,
        second element is the set of parameters that are integer, third element is the set of params that are
        multivariate, and fourth element is the set of params that are univariate
        """
        categorical_params = set()
        int_params = set()
        multivariate_params = set()
        univariate_params = set()

        for par in param_dict:
            if isinstance(param_dict[par], rv_frozen):
                # FIXME: what if the distribution generators ints , GP would convert it to float
                univariate_params.add(par)
            elif isinstance(param_dict[par], multi_rv_frozen):
                multivariate_params.add(par)

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
            else:
                warnings.warn(f"Parameter {par} could not be classified")

        return (categorical_params, int_params, multivariate_params, univariate_params)
