"""
Main Tuner Class which uses other abstractions.
General usage is to find the optimal hyperparameters of the classifier
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
import logging
import random
from typing import Callable
import warnings

from tqdm.auto import tqdm
import numpy as np

from mango.domain.domain_space import DomainSpace
from mango.optimizer.bayesian_learning import BayesianLearning
from mango.domain.parameter_sampler import parameter_sampler

# setting warnings to ignore for now
warnings.filterwarnings("ignore")

_logger = logging.getLogger(__name__)


@dataclass
class TunerConfig:
    domain_size: int = None
    initial_random: int = 2
    initial_custom: list = None
    num_iteration: int = 20
    batch_size: int = 1
    optimizer: str = "Bayesian"
    parallel_strategy: str = "clustering"
    surrogate: object = None  # used to test different kernel functions
    valid_optimizers = ["Bayesian", "Random"]
    valid_parallel_strategies = ["penalty", "clustering"]
    alpha: float = 2.0
    exploration: float = 1.0
    exploration_decay: float = 0.9
    exploration_min: float = 0.1
    fixed_domain: bool = False
    early_stopping: Callable = None
    constraint: Callable = None
    param_sampler: Callable = parameter_sampler
    scale_params: bool = False

    def __post_init__(self):
        if self.optimizer not in self.valid_optimizers:
            raise ValueError(
                f"optimizer: {self.optimizer} is not valid, should be one of {self.valid_optimizers}"
            )
        if self.parallel_strategy not in self.valid_parallel_strategies:
            raise ValueError(
                f"parallel strategy: {self.parallel_strategy} is not valid,"
                f" should be one of {self.valid_parallel_strategies}"
            )

    @property
    def is_bayesian(self):
        return self.optimizer == "Bayesian"

    @property
    def is_random(self):
        return self.optimizer == "Random"

    @property
    def strategy_is_penalty(self):
        return self.parallel_strategy == "penalty"

    @property
    def strategy_is_clustering(self):
        return self.parallel_strategy == "clustering"

    def early_stop(self, results):
        if self.early_stopping is None:
            return False

        results = copy.deepcopy(results)
        return self.early_stopping(results)


class Tuner:

    def __init__(self, param_dict, objective, conf_dict=None):

        self.objective_function = objective
        self.maximize_objective = True

        if conf_dict is None:
            conf_dict = {}

        self.config = TunerConfig(**conf_dict)
        self.ds = DomainSpace(
            param_dict,
            param_sampler=self.config.param_sampler,
            constraint=self.config.constraint,
            scale_params=self.config.scale_params,
        )

        if self.config.domain_size is not None:
            self.ds.domain_size = self.config.domain_size

        # overwrite batch size if given as a property of objective function
        if hasattr(objective, "batch_size"):
            self.config.batch_size = objective.batch_size

        # stores the results of using the tuner
        self.results = dict()

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

    def process_initial_custom(self):
        init_values = self.config.initial_custom

        if all(isinstance(v, Mapping) for v in init_values):
            X_init = copy.deepcopy(init_values)
            X_list, Y_list = self.runUserObjective(X_init)
            return X_list, Y_list
        elif all(isinstance(v, tuple) and len(v) == 2 for v in init_values):
            X_list = copy.deepcopy([v[0] for v in init_values])
            Y_list = np.array([v[1] for v in init_values])
            if self.maximize_objective is False:
                Y_list = -1 * Y_list
            return X_list, Y_list
        else:
            raise TypeError(
                f"Elements of initial_custom param should be either a dict of params or tuple (params, y),"
                f" got: {[type(v).__name__ for v in init_values]}"
            )

    def run_initial(self):
        if self.config.initial_custom is not None:
            return self.process_initial_custom()
        else:
            # getting first few random values
            X_init = self.ds.get_random_sample(self.config.initial_random)
            X_list, Y_list = self.runUserObjective(X_init)

            # in case initial random results are invalid try different samples
            n_tries = 1
            while len(Y_list) < self.config.initial_random and n_tries < 3:
                X_init = self.ds.get_random_sample(
                    self.config.initial_random - len(Y_list)
                )
                X_list2, Y_list2 = self.runUserObjective(X_init)
                X_list = np.append(X_list, X_list2)
                Y_list = np.append(Y_list, Y_list2)
                n_tries += 1

            if len(Y_list) == 0:
                raise ValueError(
                    "No valid configuration found to initiate the Bayesian Optimizer"
                )
        return X_list, Y_list

    def runBayesianOptimizer(self):
        results = dict()

        X_list, Y_list = self.run_initial()

        # evaluated hyperparameters are used
        X_init = self.ds.convert_GP_space(X_list)
        Y_init = Y_list.reshape(len(Y_list), 1)

        # setting the initial random hyperparameters tried
        results["random_params"] = X_list
        results["random_params_objective"] = Y_list

        Optimizer = BayesianLearning(
            surrogate=self.config.surrogate,
            alpha=self.config.alpha,
            domain_size=self.config.domain_size,
        )

        X_sample = X_init
        Y_sample = Y_init

        hyper_parameters_tried = X_list
        objective_function_values = Y_list
        surrogate_values = Y_list

        x_failed_evaluations = np.array([])

        domain_list = self.ds.get_domain()
        X_domain_np = self.ds.convert_GP_space(domain_list)

        # running the iterations
        pbar = tqdm(range(self.config.num_iteration))
        for _ in pbar:

            # adding a Minimum exploration to explore independent of UCB
            if random.random() < self.config.exploration:
                random_parameters = self.ds.get_random_sample(self.config.batch_size)
                X_next_batch = self.ds.convert_GP_space(random_parameters)

                if self.config.exploration > self.config.exploration_min:
                    self.config.exploration = (
                        self.config.exploration * self.config.exploration_decay
                    )

            elif self.config.strategy_is_penalty:
                X_next_batch = Optimizer.get_next_batch(
                    X_sample, Y_sample, X_domain_np, batch_size=self.config.batch_size
                )
            elif self.config.strategy_is_clustering:
                X_next_batch = Optimizer.get_next_batch_clustering(
                    X_sample, Y_sample, X_domain_np, batch_size=self.config.batch_size
                )
            else:
                # assume penalty approach
                X_next_batch = Optimizer.get_next_batch(
                    X_sample, Y_sample, X_domain_np, batch_size=self.config.batch_size
                )

            # Scheduler
            X_next_PS = self.ds.convert_PS_space(X_next_batch)

            # if all the xs have failed before, replace them with random sample
            # as we will not get any new information otherwise
            if all(x in x_failed_evaluations for x in X_next_PS):
                X_next_PS = self.ds.get_random_sample(self.config.batch_size)

            # Evaluate the Objective function
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
            X_next_batch = self.ds.convert_GP_space(X_next_list)

            # update the bookeeping of values tried
            hyper_parameters_tried = np.append(hyper_parameters_tried, X_next_list)
            objective_function_values = np.append(
                objective_function_values, Y_next_list
            )
            surrogate_values = np.append(
                surrogate_values, Optimizer.surrogate.predict(X_next_batch)
            )

            # Appending to the current samples
            X_sample = np.vstack((X_sample, X_next_batch))
            Y_sample = np.vstack((Y_sample, Y_next_batch))

            # refresh domain if not fixed
            if not self.config.fixed_domain:
                domain_list = self.ds.get_domain()
                X_domain_np = self.ds.convert_GP_space(domain_list)

            results["params_tried"] = hyper_parameters_tried
            results["objective_values"] = objective_function_values
            results["surrogate_values"] = surrogate_values

            results["best_objective"] = np.max(results["objective_values"])
            results["best_params"] = results["params_tried"][
                np.argmax(results["objective_values"])
            ]
            if self.maximize_objective is False:
                results["objective_values"] = -1 * results["objective_values"]
                results["best_objective"] = -1 * results["best_objective"]

            pbar.set_description("Best score: %s" % results["best_objective"])

            # check if early stop criteria has been met
            if self.config.early_stop(results):
                _logger.info("Early stopping criteria satisfied")
                break

        # saving the optimizer and ds in the tuner object which can save the surrogate function and ds details
        self.Optimizer = Optimizer

        return results

    def runRandomOptimizer(self):
        results = dict()

        X_sample_list = []
        Y_sample_list = []

        batch_size = self.config.batch_size
        n_iterations = self.config.num_iteration
        random_hyper_parameters = self.ds.get_random_sample(n_iterations * batch_size)

        # running the iterations
        pbar = tqdm(range(0, len(random_hyper_parameters), batch_size))
        for idx in pbar:
            # getting batch by batch random values to try
            batch_hyper_parameters = random_hyper_parameters[idx : idx + batch_size]
            X_list, Y_list = self.runUserObjective(batch_hyper_parameters)

            X_sample_list = np.append(X_sample_list, X_list)
            Y_sample_list = np.append(Y_sample_list, Y_list)

            results["params_tried"] = X_sample_list
            results["objective_values"] = Y_sample_list

            results["best_objective"] = np.max(results["objective_values"])
            results["best_params"] = results["params_tried"][
                np.argmax(results["objective_values"])
            ]
            if self.maximize_objective is False:
                results["objective_values"] = -1 * results["objective_values"]
                results["best_objective"] = -1 * results["best_objective"]

            pbar.set_description("Best score: %s" % results["best_objective"])

            # check if early stop criteria has been met
            if self.config.early_stop(results):
                _logger.info("Early stopping criteria satisfied")
                break

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
