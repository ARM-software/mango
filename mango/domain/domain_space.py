"""
Define the domain space abstractions for the Optimizer
"""

import math
import numpy as np
from collections.abc import Callable
import warnings
from itertools import compress

from .parameter_sampler import AbstractParameterSampler, ParameterSampler


class DomainSpace:
    def __init__(
        self,
        param_sampler: type[AbstractParameterSampler] = ParameterSampler,
        constraint: Callable = None,
        constraint_max_retries: int = 10,
    ):
        self.param_sampler = param_sampler
        self.constraint = constraint
        self.constraint_max_tries = constraint_max_retries

    def get_domain(self):
        return self.get_random_sample(self.param_sampler.domain_size)

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
        return self.param_sampler.sample(size)

    def convert_GP_space(self, domain_list):
        """
        convert the hyperparameters from the param_dict space to the GP space, by converting the
        categorical variables to one hotencoded, and return a numpy array which can be used to train the GP

        input is the domain_list generated using the Parameter Sampler.
        """
        categorical_params = self.param_sampler.categorical_params

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
                    size = len(
                        self.param_sampler.dist[x]
                    )  # total number of categories.
                    # we need to see the index where domain[x] appears in param_dict[x]
                    index = self.param_sampler.categorical_index_lookup[x][domain[x]]
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

        categorical_params = self.param_sampler.categorical_params
        int_params = self.param_sampler.integer_params
        param_dict = self.param_sampler.dist

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
