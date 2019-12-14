"""
Main Tuner Class which uses other abstractions.
Genereal usage is to find the optimal hyper-parameters of the classifier
"""

from mango.domain.domain_space import domain_space
from mango.optimizer.bayesian_learning import BayesianLearning
from scipy.stats._distn_infrastructure import rv_frozen

import numpy as np

##setting warnings to ignore for now
import warnings

warnings.filterwarnings('ignore')


class Tuner():
    def __init__(self, param_dict, objective, conf_dict=None):
        # stores the configuration used by the tuner
        self.conf_Dict = dict()

        # stores the results of using the tuner
        self.maximize_objective = True
        self.results = dict()

        # param_dict is a required parameter
        self.conf_Dict['param_dict'] = param_dict

        # Objective funtion is a required parameter
        self.conf_Dict['userObjective'] = objective

        self.conf_Dict['domain_size'] = None
        self.conf_Dict['initial_random'] = 1
        self.conf_Dict['num_iteration'] = 20
        self.conf_Dict['objective'] = "maximize"  # only maximize is allowed
        self.conf_Dict['batch_size'] = 1

        # setting default optimizer to Bayesian
        self.conf_Dict['optimizer'] = "Bayesian"

        # in case the optional conf_dict is passed
        if conf_dict != None:
            if 'objective' in conf_dict:
                self.conf_Dict['objective'] = conf_dict['objective']

            if 'num_iteration' in conf_dict:
                self.conf_Dict['num_iteration'] = conf_dict['num_iteration']

            if 'domain_size' in conf_dict:
                self.conf_Dict['domain_size'] = conf_dict['domain_size']

            if 'initial_random' in conf_dict:
                if conf_dict['initial_random'] > 0:
                    self.conf_Dict['initial_random'] = conf_dict['initial_random']

            if 'batch_size' in conf_dict:
                if conf_dict['batch_size'] > 0:
                    self.conf_Dict['batch_size'] = conf_dict['batch_size']

            if 'optimizer' in conf_dict:
                self.conf_Dict['optimizer'] = conf_dict['optimizer']

            if 'surrogate' in conf_dict:
                self.conf_Dict['surrogate'] = conf_dict['surrogate']

        # Calculating the domain size based on the param_dict
        if self.conf_Dict['domain_size'] == None:
            self.calculateDomainSize()

        # overwrite batch size if given as a property of objective function
        # see schedulers
        if hasattr(objective, 'batch_size'):
            self.conf_Dict['batch_size'] = objective.batch_size


    """
    Calculating the domain size to be explored for finding
    optimum of bayesian optimizer
    """

    def calculateDomainSize(self):
        # Minimum and maximum domain size
        domain_min = 5000
        domain_max = 50000

        param_dict = self.conf_Dict['param_dict']
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

        # print('Calculated Domain Size:',domain_size)
        if domain_size < domain_min:
            domain_size = domain_min

        if domain_size > domain_max:
            domain_size = domain_max

        self.conf_Dict['domain_size'] = domain_size

    def getConf(self):
        return self.conf_Dict

    def run(self):
        # running the optimizer
        if self.conf_Dict['optimizer'] == "Bayesian":
            self.results = self.runBayesianOptimizer()
        elif self.conf_Dict['optimizer'] == "Random":
            self.results = self.runRandomOptimizer()
        else:
            print("Error: Unknowm Optimizer")
        return self.results

    """
    Main function used by tuner to run the classifier evaluation
    """

    def maximize(self):
        return self.run()

    def minimize(self):
        self.maximize_objective = False
        return self.run()

    """
    - Called by runLocal.
    - Used to locally evaluate the classifier
    - calls run_clf_local function which is available in scheduler.LocalTasks
    """

    def runBatchLocal(self, X_batch_list):
        results = []
        for hyper_par in X_batch_list:
            result = run_clf_local(self.conf_Dict['clf_name'], self.conf_Dict['dataset_name'], hyper_par)
            results.append(result)
        return np.array(results).reshape(len(results), 1), results

    def runBayesianOptimizer(self):
        results = dict()
        # domain space abstraction
        ds = domain_space(self.conf_Dict['param_dict'], self.conf_Dict['domain_size'])

        # getting first few random values
        random_hyper_parameters = ds.get_random_sample(self.conf_Dict['initial_random'])
        X_list, Y_list = self.runUserObjective(random_hyper_parameters)

        # in case initial random results are invalid try different samples
        n_tries = 1
        while len(Y_list) < self.conf_Dict['initial_random'] and n_tries < 3:
            random_hps = ds.get_random_sample(self.conf_Dict['initial_random'] - len(Y_list))
            X_list2, Y_list2 = self.runUserObjective(random_hps)
            random_hyper_parameters.extend(random_hps)
            X_list = np.append(X_list, X_list2)
            Y_list = np.append(Y_list, Y_list2)
            n_tries += 1

        if len(Y_list) == 0:
            raise ValueError("No valid configuration found to initiate the Bayesian Optimizer")

        # evaluated hyper parameters are used
        X_init = ds.convert_GP_space(X_list)
        Y_init = Y_list.reshape(len(Y_list), 1)

        # setting the initial random hyper parameters tried
        results['random_params'] = X_list
        results['random_params_objective'] = Y_list

        Optimizer = BayesianLearning(surrogate=self.conf_Dict.get('surrogate'))
        Optimizer.domain_size = self.conf_Dict['domain_size']

        X_sample = X_init
        Y_sample = Y_init

        hyper_parameters_tried = random_hyper_parameters
        objective_function_values = Y_list

        # running the iterations
        for i in range(self.conf_Dict['num_iteration']):
            # Domain Space
            domain_list = ds.get_domain()
            X_domain_np = ds.convert_GP_space(domain_list)

            # Black-Box Optimizer
            X_next_batch = Optimizer.get_next_batch(X_sample, Y_sample, X_domain_np,
                                                    batch_size=self.conf_Dict['batch_size'])
            # X_next_batch = Optimizer.get_next_batch_clustering(X_sample,Y_sample,X_domain_np,batch_size=self.conf_Dict['batch_size'])

            # Scheduler
            X_next_PS = ds.convert_PS_space(X_next_batch)

            # Evaluate the Objective function
            # Y_next_batch, Y_next_list = self.runUserObjective(X_next_PS)
            X_next_list, Y_next_list = self.runUserObjective(X_next_PS)
            Y_next_batch = Y_next_list.reshape(len(Y_next_list), 1)
            # update X_next_batch to successfully evaluated values
            X_next_batch = ds.convert_GP_space(X_next_list)

            # update the bookeeping of values tried
            hyper_parameters_tried = np.append(hyper_parameters_tried, X_next_list)
            objective_function_values = np.append(objective_function_values, Y_next_list)

            # Appending to the current samples
            X_sample = np.vstack((X_sample, X_next_batch))
            Y_sample = np.vstack((Y_sample, Y_next_batch))

        results['params_tried'] = hyper_parameters_tried
        results['objective_values'] = objective_function_values

        results['best_objective'] = np.max(Y_sample)
        results['best_params'] = hyper_parameters_tried[np.argmax(Y_sample)]

        # saving the optimizer and ds in the tuner object which can save the surrogate function and ds details
        self.Optimizer = Optimizer
        self.ds = ds
        return results

    def runRandomOptimizer(self):
        results = dict()
        # domain space abstraction
        ds = domain_space(self.conf_Dict['param_dict'], self.conf_Dict['domain_size'])

        X_sample_list = []
        Y_sample_list = []

        # running the iterations
        for i in range(self.conf_Dict['num_iteration']):
            # getting batch by batch random values to try
            random_hyper_parameters = ds.get_random_sample(self.conf_Dict['batch_size'])
            X_list, Y_list = self.runUserObjective(random_hyper_parameters)

            X_sample_list = np.append(X_sample_list, X_list)
            Y_sample_list = np.append(Y_sample_list, Y_list)

        # After all the iterations are done now bookeeping and best hyper parameter values
        results['params_tried'] = X_sample_list
        results['objective_values'] = Y_sample_list

        if len(Y_sample_list) > 0:
            results['best_objective'] = np.max(Y_sample_list)
            results['best_params'] = X_sample_list[np.argmax(Y_sample_list)]

        return results

    def runUserObjective(self, X_next_PS):

        # initially assuming entire X_next_PS is evaluated and returned results are only Y values
        X_list_evaluated = X_next_PS
        results = self.conf_Dict['userObjective'](X_next_PS)
        Y_list_evaluated = results

        """
        if result is a tuple, then there is possibility that partial values are evaluated
        """
        if isinstance(results, tuple):
            X_list_evaluated, Y_list_evaluated = results
            # return np.array(Y_list_evaluated).reshape(len(Y_list_evaluated),1),
        # return np.array(results).reshape(len(results),1),results

        X_list_evaluated = np.array(X_list_evaluated)
        if self.maximize_objective is False:
            Y_list_evaluated = -1 * np.array(Y_list_evaluated)
        else:
            Y_list_evaluated = np.array(Y_list_evaluated)

        return X_list_evaluated, Y_list_evaluated
