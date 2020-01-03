"""
Main Tuner Class which uses other abstractions.
General usage is to find the optimal hyper-parameters of the classifier
"""

from dataclasses import dataclass

from mango.domain.domain_space import domain_space
from mango.optimizer.bayesian_learning import BayesianLearning
from scipy.stats._distn_infrastructure import rv_frozen

from tqdm.auto import tqdm
import numpy as np

## setting warnings to ignore for now
import warnings

warnings.filterwarnings('ignore')


class Tuner:

    @dataclass
    class Config:
        domain_size: int = None
        initial_random: int = 1
        num_iteration: int = 20
        batch_size: int = 1
        optimizer: str = 'Bayesian'
        parallel_strategy: str = 'penalty'
        surrogate: object = None # used to test different kernel functions
        scale_y: bool = False

        valid_optimizers = ['Bayesian', 'Random']
        valid_parallel_strategies = ['penalty', 'clustering']

        def __post_init__(self):
            if self.optimizer not in self.valid_optimizers:
                raise ValueError(f'optimizer: {self.optimizer} is not valid, should be one of {self.valid_optmizers}')
            if self.parallel_strategy not in self.valid_parallel_strategies:
                raise ValueError(f'parallel strategy: {self.parallel_strategy} is not valid, should be one of {self.valid_parallel_strategies}')

        @property
        def is_bayesian(self):
            return self.optimizer == 'Bayesian'

        @property
        def is_random(self):
            return self.optimizer == 'Random'

        @property
        def strategy_is_penalty(self):
            return self.parallel_strategy == 'penalty'

        @property
        def strategy_is_clustering(self):
            return self.parallel_strategy == 'clustering'

    def __init__(self, param_dict, objective, conf_dict=None):

        self.param_dict = param_dict
        self.objective_function = objective
        self.maximize_objective = True

        if conf_dict is None:
            conf_dict = {}

        self.config = Tuner.Config(**conf_dict)

        if self.config.domain_size is None:
            self.config.domain_size = self.calculateDomainSize(self.param_dict)

        # overwrite batch size if given as a property of objective function
        if hasattr(objective, 'batch_size'):
            self.config.batch_size = objective.batch_size

        # stores the results of using the tuner
        self.results = dict()

    @staticmethod
    def calculateDomainSize(param_dict):
        """
           Calculating the domain size to be explored for finding
           optimum of bayesian optimizer
        """
        # Minimum and maximum domain size
        domain_min = 5000
        domain_max = 500000

        domain_size = 1

        for par in param_dict:
            if isinstance(param_dict[par], rv_frozen):
                distrib = param_dict[par]
                loc, scale = distrib.args
                min_scale = 1
                scale = int(scale)
                if scale < min_scale:
                    scale = min_scale

                domain_size = domain_size * scale * 50

            elif isinstance(param_dict[par], range):
                domain_size = domain_size * len(param_dict[par])

            elif isinstance(param_dict[par], list):
                domain_size = domain_size * len(param_dict[par])

        if domain_size < domain_min:
            domain_size = domain_min

        if domain_size > domain_max:
            domain_size = domain_max

        return domain_size

    def run(self):
        if self.config.is_bayesian:
            self.results = self.runBayesianOptimizer()
        elif self.config.is_random:
            self.results = self.runRandomOptimizer()
        else:
            raise ValueError("Unknown Optimizer %s" % self.config.optimizer)

        return self.results

    def maximize(self):
        return self.run()

    def minimize(self):
        self.maximize_objective = False
        return self.run()

    def runBayesianOptimizer(self):
        results = dict()

        # domain space abstraction
        ds = domain_space(self.param_dict, self.config.domain_size)

        # getting first few random values
        random_hyper_parameters = ds.get_random_sample(self.config.initial_random)
        X_list, Y_list = self.runUserObjective(random_hyper_parameters)

        # in case initial random results are invalid try different samples
        n_tries = 1
        while len(Y_list) < self.config.initial_random and n_tries < 3:
            random_hps = ds.get_random_sample(self.config.initial_random - len(Y_list))
            X_list2, Y_list2 = self.runUserObjective(random_hps)
            random_hyper_parameters.extend(random_hps)
            X_list = np.append(X_list, X_list2)
            Y_list = np.append(Y_list, Y_list2)
            n_tries += 1

        if len(Y_list) == 0:
            raise ValueError("No valid configuration found to initiate the Bayesian Optimizer")

        # evaluated hyper parameters are used
        X_init = ds.convert_to_gp(X_list)
        Y_init = Y_list.reshape(len(Y_list), 1)

        # setting the initial random hyper parameters tried
        results['random_params'] = X_list
        results['random_params_objective'] = Y_list

        Optimizer = BayesianLearning(surrogate=self.config.surrogate, n_features=X_init.shape[1])
        Optimizer.domain_size = self.config.domain_size

        X_sample = X_init
        Y_sample = Y_init

        hyper_parameters_tried = random_hyper_parameters
        objective_function_values = Y_list
        surrogate_values = Y_list

        x_failed_evaluations = np.array([])

        # running the iterations
        pbar = tqdm(range(self.config.num_iteration))
        for i in pbar:
            # Domain Space
            X_domain_np = ds.sample_gp_space()

            # Black-Box Optimizer
            if self.config.scale_y:
                scale = np.max(Y_sample) - np.min(Y_sample)
                if scale == 0:
                    scale = np.max(np.abs(Y_sample))
                Y_scaled = (Y_sample - np.mean(Y_sample)) / scale
            else:
                Y_scaled = Y_sample

            if self.config.strategy_is_penalty:
                X_next_batch = Optimizer.get_next_batch(X_sample, Y_scaled, X_domain_np,
                                                    batch_size=self.config.batch_size)
            elif self.config.strategy_is_clustering:
                X_next_batch = Optimizer.get_next_batch_clustering(X_sample,Y_scaled, X_domain_np,
                                                                   batch_size=self.config.batch_size)
            else:
                # assume penalty approach
                X_next_batch = Optimizer.get_next_batch(X_sample, Y_scaled, X_domain_np,
                                                        batch_size=self.config.batch_size)

            # Scheduler
            X_next_PS = ds.convert_to_params(X_next_batch)

            # if all the xs have failed before, replace them with random sample
            # as we will not get any new information otherwise
            if all(x in x_failed_evaluations for x in X_next_PS):
                X_next_PS = ds.get_random_sample(self.config.batch_size)

            # Evaluate the Objective function
            # Y_next_batch, Y_next_list = self.runUserObjective(X_next_PS)
            X_next_list, Y_next_list = self.runUserObjective(X_next_PS)

            # keep track of all parameters that failed
            x_failed = [x for x in X_next_PS if x not in X_next_list]
            x_failed_evaluations = np.append(x_failed_evaluations, x_failed)

            if len(Y_next_list) == 0:
                # no values returned
                # this is problematic if domain is small and same value is tried again in the next iteration as the optimizer would be stuck
                continue

            Y_next_batch = Y_next_list.reshape(len(Y_next_list), 1)
            # update X_next_batch to successfully evaluated values
            X_next_batch = ds.convert_to_gp(X_next_list)

            # update the bookeeping of values tried
            hyper_parameters_tried = np.append(hyper_parameters_tried, X_next_list)
            objective_function_values = np.append(objective_function_values, Y_next_list)
            surrogate_values = np.append(surrogate_values, Optimizer.surrogate.predict(X_next_batch))

            # Appending to the current samples
            X_sample = np.vstack((X_sample, X_next_batch))
            Y_sample = np.vstack((Y_sample, Y_next_batch))
            pbar.set_description("Best score: %s" % np.max(Y_sample))

        results['params_tried'] = hyper_parameters_tried
        results['objective_values'] = objective_function_values
        results['surrogate_values'] = surrogate_values

        results['best_objective'] = np.max(Y_sample)
        results['best_params'] = hyper_parameters_tried[np.argmax(Y_sample)]

        if self.maximize_objective is False:
            results['objective_values'] = -1 * results['objective_values']
            results['best_objective'] = -1 * results['best_objective']

        # saving the optimizer and ds in the tuner object which can save the surrogate function and ds details
        self.Optimizer = Optimizer
        self.ds = ds
        return results

    def runRandomOptimizer(self):
        results = dict()
        # domain space abstraction
        ds = domain_space(self.param_dict, self.config.domain_size)

        X_sample_list = []
        Y_sample_list = []

        # running the iterations
        pbar = tqdm(range(self.config.num_iteration))
        for i in pbar:
            # getting batch by batch random values to try
            random_hyper_parameters = ds.get_random_sample(self.config.batch_size)
            X_list, Y_list = self.runUserObjective(random_hyper_parameters)

            X_sample_list = np.append(X_sample_list, X_list)
            Y_sample_list = np.append(Y_sample_list, Y_list)

            pbar.set_description("Best score: %s" % np.max(np.array(Y_sample_list)))

        # After all the iterations are done now bookkeeping and best hyper parameter values
        results['params_tried'] = X_sample_list
        results['objective_values'] = Y_sample_list

        if len(Y_sample_list) > 0:
            results['best_objective'] = np.max(Y_sample_list)
            results['best_params'] = X_sample_list[np.argmax(Y_sample_list)]

        if self.maximize_objective is False:
            results['objective_values'] = -1 * results['objective_values']
            results['best_objective'] = -1 * results['best_objective']

        return results

    def runUserObjective(self, X_next_PS):
        # initially assuming entire X_next_PS is evaluated and returned results are only Y values
        X_list_evaluated = X_next_PS
        results = self.objective_function(X_next_PS)
        Y_list_evaluated = results

        # if result is a tuple, then there is possibility that partial values are evaluated
        if isinstance(results, tuple):
            X_list_evaluated, Y_list_evaluated = results

        X_list_evaluated = np.array(X_list_evaluated)
        if self.maximize_objective is False:
            Y_list_evaluated = -1 * np.array(Y_list_evaluated)
        else:
            Y_list_evaluated = np.array(Y_list_evaluated)

        return X_list_evaluated, Y_list_evaluated
