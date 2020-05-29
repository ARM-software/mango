"""
Define the domain space abstractions for the Optimizer
"""

# this is used to check whether the domain space object is a of type rv_frozen.
# if the parameter type is a disribution or rv_frozen, the parameter sampler can handle it
from scipy.stats._distn_infrastructure import rv_frozen
import numpy as np
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler
from collections.abc import Iterable

class domain_space():
    """
     initializer
     1) We expect a parameter dictionary as an input from which we will creating a mapping.
     2) The mapping is storing the categorical variables or discrete variables along with the discrete possible values
     3) The categorical variables are handeled as one hot encoding values
    """

    def __init__(self,
                 param_dict,
                 domain_size,
                 scaled=False):
        self.param_dict = param_dict

        # the domain size to explore using the parameter sampler
        self.domain_size = domain_size

        # creating a mapping of categorical variables
        self.create_mappings()

        # create scaler for GP features
        # only used with old sampling methods
        if scaled:
            self.scaler = self.create_scaler()

    def create_scaler(self):
        """
        create scaler for features that go into GPR
        using a fixed set of initial domain samples
        """
        domain_list = self.get_domain()
        x_gp = self.convert_GP_space(domain_list)
        scaler = StandardScaler()
        scaler.fit(x_gp)
        return scaler


    """
    returns the list of domain values using the parameter sampler
    The size of values is the domain_size
    """

    def get_domain(self):
        domain_list = list(ParameterSampler(self.param_dict, n_iter=self.domain_size))
        return domain_list

    """
    return a random sample using the ParameterSample
    """

    def get_random_sample(self, size):
        domain_list = list(ParameterSampler(self.param_dict, n_iter=size))
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
        intervals = dict()

        param_dict = self.param_dict
        for par in param_dict:
            if isinstance(param_dict[par], rv_frozen):
                # FIXME: what if the distribution generators ints , GP would convert it to float
                intervals[par] = param_dict[par].interval(1.)
                pass  # we are not doing anything at present, and will directly use its value for GP.

            elif isinstance(param_dict[par], range):
                mapping_int[par] = param_dict[par]
                intervals[par] = (min(param_dict[par]), max(param_dict[par]))

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
                    intervals[par] = (min(param_dict[par]), max(param_dict[par]))

                # For lists with mixed type, floats or strings we consider them categorical or discrete
                else:
                    mapping_categorical[par] = param_dict[par]
                    intervals[par] = (0, 1)

        self.mapping_categorical = mapping_categorical
        self.mapping_int = mapping_int
        self.intervals = intervals

    @property
    def gp_features_count(self):
        m = 0
        for param in self.param_dict:
            if param in self.mapping_categorical:
                m += len(self.mapping_categorical[param])
            else:
                m += 1

        return m

    @property
    def param_gp_index(self):
        """
        Returns dict mapping parameter name to gp feature index
        """
        res = {}
        idx = 0
        for param in sorted(self.param_dict.keys()):
            if param not in self.mapping_categorical:
                res[param] = idx
                idx += 1
            else:
                res[param] = idx
                idx += len(self.mapping_categorical[param])

        return res

    def sample_gp_space(self, n_samples=None):
        """
        Generate sample in the Gaussian process space

        Returns a n_samples x m numpy array (m is the number of GP features)
        """
        if n_samples is None:
            n_samples = self.domain_size

        m = self.gp_features_count
        idx_map = self.param_gp_index
        X = np.zeros((n_samples, m))
        for param in self.param_dict:
            distribution = self.param_dict[param]
            if param not in self.mapping_categorical:
                if isinstance(distribution, rv_frozen):
                    X[:, idx_map[param]] = np.random.uniform(size=n_samples)
                else:
                    random_samples = np.random.choice(distribution, size=n_samples)
                    _min = self.intervals[param][0]
                    _max = self.intervals[param][1]
                    _scale = _max - _min
                    if _max == _min:
                        _scale = 1.
                    X[:, idx_map[param]] = (random_samples - _min) / _scale
            else:
                n_classes = len(self.mapping_categorical[param])
                random_one_hot = np.eye(n_classes)[np.random.choice(n_classes, n_samples)]
                X[:, idx_map[param]:idx_map[param] + n_classes] = random_one_hot

        return X

    def convert_to_params(self, X):
        """
        Convert GP feature samples to Parameter dictionaries (inverse of sample_gp_space method)
        Params:
        X: n_samples x m_gp_features numpy array

        Returns list of dictionaries with param names as keys
        """
        res = {}
        idx_map = self.param_gp_index
        for param in self.param_dict:
            gp_index = idx_map[param]
            distribution = self.param_dict[param]
            if param not in self.mapping_categorical:
                if isinstance(distribution, rv_frozen):
                    val = distribution.ppf(X[:, gp_index])
                else:
                    _min = self.intervals[param][0]
                    _max = self.intervals[param][1]
                    _scale = _max - _min
                    if _max == _min:
                        _scale = 1.
                    val = (X[:, gp_index] * _scale) + _min

                if param in self.mapping_int:
                    val = [int(v) for v in np.rint(val)]

                res[param] = val
            else:
                n_classes = len(self.mapping_categorical[param])
                one_hot_samples = X[:, gp_index:gp_index + n_classes]
                res[param] = [distribution[idx] for idx in np.argmax(one_hot_samples, axis=1)]

        # convert dict of lists to list of dicts
        result = []
        for idx in range(X.shape[0]):
            result.append({param: value[idx] for param, value in res.items()})

        return result

    def convert_to_gp(self, params_list):
        """
        Convert parameters to GP features

        params_list: list of dicts with parameter name as keys

        Returns 2D array n_samples x m_gp_features
        """
        n_samples = len(params_list)
        m = self.gp_features_count
        X = np.zeros((n_samples, m))
        idx_map = self.param_gp_index

        for param in self.param_dict:
            gp_index = idx_map[param]
            values = [d[param] for d in params_list]
            distribution = self.param_dict[param]
            if param not in self.mapping_categorical:
                values = np.array(values)
                if isinstance(distribution, rv_frozen):
                    X[:, gp_index] = distribution.cdf(values)
                else:
                    _min = self.intervals[param][0]
                    _max = self.intervals[param][1]
                    _scale = _max - _min
                    if _max == _min:
                        _scale = 1.
                    X[:, gp_index] = (values - _min) / _scale

            else:
                n_classes = len(self.mapping_categorical[param])
                one_hot = np.zeros((n_samples, n_classes))
                for row, v in enumerate(values):
                    one_hot[row, distribution.index(v)] = 1
                X[:, gp_index:gp_index + n_classes] = one_hot

        return X

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
        if hasattr(self, 'scaler'):
            X = self.scaler.transform(X)

        return X

    """
    Convert from the X_gp space which is a numpy array that can be given input to the gaussian process
    to the parameter sampler space which is a list of the dict

    We have to reverse the one-hotencoded transformation of the categories to the category name
    """

    def convert_PS_space(self, X_gp):
        X_ps = []

        if hasattr(self, 'scaler'):
            X_gp = self.scaler.inverse_transform(X_gp)

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
