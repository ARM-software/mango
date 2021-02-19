"""
Define the domain space abstractions for the Optimizer
"""

# this is used to check whether the domain space object is a of type rv_frozen.
# if the parameter type is a disribution or rv_frozen, the parameter sampler can handle it
from scipy.stats._distn_infrastructure import rv_frozen
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections.abc import Iterable

from .batch_parameter_sampler import BatchParameterSampler


class domain_space():
    """
     initializer
     1) We expect a parameter dictionary as an input from which we will creating a mapping.
     2) The mapping is storing the categorical variables or discrete variables along with the discrete possible values
     3) The categorical variables are handeled as one hot encoding values
    """

    def __init__(self,
                 param_dict,
                 domain_size):
        self.param_dict = param_dict

        # the domain size to explore using the parameter sampler
        self.domain_size = domain_size

        # creating a mapping of categorical variables
        self.create_mappings()


    """
    returns the list of domain values using the parameter sampler
    The size of values is the domain_size
    """

    def get_domain(self):
        domain_list = list(BatchParameterSampler(self.param_dict, n_iter=self.domain_size))
        return domain_list

    """
    return a random sample using the ParameterSample
    """

    def get_random_sample(self, size):
        domain_list = list(BatchParameterSampler(self.param_dict, n_iter=size))
        return domain_list

    """
    Categorical values/discrete values are considered from the list of each value being str

    We will finally do one-hot-encoding of the list values.
    Here we will keep some book-keeping information, like number of different values
    and the mapping of each

    Integer values are considered from list of each value as int or from a range
    """

    def create_mappings(self):
        mapping_categorical = dict()
        mapping_int = dict()

        param_dict = self.param_dict
        for par in param_dict:
            if isinstance(param_dict[par], rv_frozen):
                # FIXME: what if the distribution generators ints , GP would convert it to float
                pass  # we are not doing anything at present, and will directly use its value for GP.

            elif isinstance(param_dict[par], range):
                mapping_int[par] = param_dict[par]

            elif isinstance(param_dict[par], Iterable):

                # for list with all int we are considering it as non-categorical
                try:
                    # this check takes care of numpy ints as well
                    all_int = all(x == int(x) and type(x) != bool
                                  for x in param_dict[par])
                except (ValueError, TypeError):
                    all_int = False

                if all_int:
                    mapping_int[par] = param_dict[par]

                # For lists with mixed type, floats or strings we consider them categorical or discrete
                else:
                    mapping_categorical[par] = param_dict[par]

        self.mapping_categorical = mapping_categorical
        self.mapping_int = mapping_int


    """
    convert the hyperparameters from the param_dict space to the GP space, by converting the
    categorical variables to one hotencoded, and return a numpy array which can be used to train the GP

    input is the domain_list generated using the Parameter Sampler.
    """

    def convert_GP_space(self, domain_list):
        mapping_categorical = self.mapping_categorical

        X = []
        for domain in domain_list:
            curr_x = []
            # for x in domain:
            for x in sorted(domain.keys()):
                # this value can be directly used, for int too, we will consider it as a float for GP
                if x not in mapping_categorical:
                    curr_x.append(domain[x])

                # this is a categorical variable which require special handling
                elif x in mapping_categorical:

                    size = len(mapping_categorical[x])  # total number of categories.

                    # we need to see the index where: domain[x] appears in mapping[x]
                    index = mapping_categorical[x].index(domain[x])

                    listofzeros = [0.0] * size

                    # We will set the value to one for one hot encoding
                    listofzeros[index] = 1.0

                    # expanding current list
                    curr_x = curr_x + listofzeros

            X.append(curr_x)

        X = np.array(X)

        return X

    """
    Convert from the X_gp space which is a numpy array that can be given input to the gaussian process
    to the parameter sampler space which is a list of the dict

    We have to reverse the one-hotencoded transformation of the categories to the category name
    """

    def convert_PS_space(self, X_gp):
        X_ps = []

        mapping_categorical = self.mapping_categorical
        mapping_int = self.mapping_int
        param_dict = self.param_dict

        for i in range(X_gp.shape[0]):

            curr_x_gp = X_gp[i]
            # we will create a list of dict, same as parameter samplers
            curr_x_ps = dict()

            index = 0

            # every sample is from the param_dict

            for par in sorted(param_dict.keys()):
                # for par in param_dict:
                # print('par is:',par)

                # this has to have integer values
                if par in mapping_int:
                    curr_x_ps[par] = int(round(curr_x_gp[index]))
                    index = index + 1

                # this par is a categorical variable and we need to handle it carefully
                elif par in mapping_categorical:
                    size = len(mapping_categorical[par])  # total number of categories.

                    one_hot_encoded = curr_x_gp[index:index + size]
                    category_type = np.argmax(one_hot_encoded)

                    category_type = mapping_categorical[par][category_type]

                    curr_x_ps[par] = category_type

                    # we have processed the entire one-hotencoded
                    index = index + size

                # this is a float value
                else:
                    curr_x_ps[par] = curr_x_gp[index]
                    index = index + 1

            X_ps.append(curr_x_ps)

        return X_ps
