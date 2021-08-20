"""
Meta Tuner Class: Used to optimize across a set of models:
- selecting intelligently the order of functions to optimize
Current implementation: Bare Metal functionality for testing.
ToDo: Improve code with better config management and remove hardcoded parameters
"""
from dataclasses import dataclass
from mango.domain.domain_space import domain_space
from mango.optimizer.bayesian_learning import BayesianLearning
from scipy.stats._distn_infrastructure import rv_frozen
import numpy as np
from tqdm.auto import tqdm
import random


## setting warnings to ignore for now
import warnings
warnings.filterwarnings('ignore')

class MetaTuner:
    @dataclass
    class Config:
        n_iter: int = 20
        n_init: int = 2

    def __init__(self,
                 param_dict_list,
                 objective_list,
                 **kwargs):
        self.param_dict_list = param_dict_list
        self.objective_list = objective_list
        self.config = MetaTuner.Config(**kwargs)
        # adjustable parameters
        self.num_of_iterations = self.config.n_iter
        self.initial_random = self.config.n_init

        #list of GPR for each objective
        self.gpr_list = []

        # store the results of MetaTuner
        self.results = dict()

        #batch size of the entire metaTuner
        self.batch_size = 1

        #batch size per function to select
        self.obj_batch_size = 1



        #use to see the info and fix the seeds metaTuner is running
        self.debug = False

        self.surrogate = None # used to test different kernel functions

        #stores the index of evaluated objectives
        self.objectives_evaluated = []

        #stores the values of objectives in the order they are evaluated
        self.objective_values_list = []

        self.seed = 0

        #rate of decay of exploration probability
        self.decay_rate = 0.90

        #The initial exploration probability
        self.exploration_rate = 1.0
        self.exploration_rate_min = 0.1

        #record which classifier/gp was used last
        self.last_used = 0

        self.store_domain = {}
        self.first_itr = {}

    def run(self):
        #run the metaTuner
        self.results = self.runExponentialTuner()
        return self.results

    @staticmethod
    def calculateDomainSize(param_dict):
        """
           Calculating the domain size to be explored for finding
           optimum of bayesian optimizer
        """
        # Minimum and maximum domain size
        domain_min = 5000
        domain_max = 30000

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

    #return the max value of Y from current evaluations
    def get_max_y_value(self, Y_dict_array):
        values = Y_dict_array[0]

        for k in range(len(self.objective_list) - 1):
            values = np.vstack((values,Y_dict_array[k+1]))

        max_val = np.max(values)

        return max_val

    def runExponentialTuner(self):
        """
        Steps:
        1-Create DS obj for each obj of param_dict_list
        2-Sample randomly from the DS objects and evaluate the objective functions.
        3- Now use the GPR for each objective to select next batch
        4- Select the best values based on fxn(surrogate) from GPR.
        5- Modify the surrogate selections so as to avoid getting struck.
        """

        results = dict()

        #random params tried
        results['random_params'] = []

        #random objective values
        results['random_params_objective'] = []

        #random objective function id
        results['random_objective_fid'] = []

        results['params_tried'] = []
        results['objective_values'] = []
        results['objective_fid'] = []


        #print('*** Entering Metatuner *** ')
        num_of_random = self.initial_random

        ds = []
        for i in self.param_dict_list:
            domain_size = self.calculateDomainSize(i)
            ds.append(domain_space(i,domain_size))

        #dict of list for each obj function
        X_dict_list = {}
        Y_dict_list = {}

        #dict of array for each obj function
        X_dict_array = {}
        Y_dict_array = {}

        #stores the maximum value of the objective function for each objective
        Y_dict_array_max = {}

        #randomly evaluate the initial points for objectives
        for i in range(len(self.param_dict_list)):

            #used later
            self.first_itr[i] = True

            if self.debug:
                #fixing random seeds to reproduce the results for debugging
                np.random.seed(self.seed)
                random.seed(self.seed)

            ds_i = ds[i]

            #make sure not all values in y are same
            done = False

            #count no of times random repeated
            num_times = 0
            while not done:
                random_hyper_parameters = ds_i.get_random_sample(num_of_random)
                y_list = self.objective_list[i](random_hyper_parameters)
                #print(i, random_hyper_parameters, y_list)
                num_times = num_times + 1

                #10 times done still there is a tie
                if num_times ==10:
                    done = True
                    y_list[0] = y_list[0] - 0.001

                for ind in range(num_of_random):
                    if y_list[0]!=y_list[ind]:
                        done = True

            #print('Random_hyper_parameters:',random_hyper_parameters, y_list)
            x_list = random_hyper_parameters

            #print(i, random_hyper_parameters, x_list, y_list)

            X_dict_list[i] =[]
            X_dict_list[i].append(x_list)

            Y_dict_list[i] =[]
            Y_dict_list[i].append(y_list)

            self.objective_values_list += y_list

            results['random_params'] = results['random_params'] + x_list
            results['random_params_objective'] = results['random_params_objective'] + y_list
            results['random_objective_fid'] = results['random_objective_fid'] + [i]*len(x_list)


            #x_array2 = ds_i.convert_GP_space(random_hyper_parameters)
            #x_array = ds_i.convert_to_gp(random_hyper_parameters)

            x_array = ds_i.convert_GP_space(random_hyper_parameters)

            X_dict_array[i] = x_array

            #print(x_array2)
            #print(x_array)

            y_array = np.array(y_list).reshape(len(y_list),1)
            Y_dict_array[i] = y_array

            #the random ones are added as it is
            Y_dict_array_max[i] = y_array

        #Initialize the number of Optimizers
        Optimizer_list = []

        for i in range(len(self.objective_list)):
            Optimizer_i = BayesianLearning(surrogate=self.surrogate)
            Optimizer_i.domain_size = ds[i].domain_size
            Optimizer_list.append(Optimizer_i)

            #print('Domain Size:',ds[i].domain_size)

        # Storing optimizers attributes: iteration count and modified exploration of external surrogate
        Optimizer_exploration = []
        Optimizer_iteration = []
        for i in range(len(self.objective_list)):
            Optimizer_exploration.append(1.0)
            Optimizer_iteration.append(1.0)

        results['params_tried'] = results['random_params']
        results['objective_values'] = results['random_params_objective']
        results['objective_fid'] = results['random_objective_fid']

        #print(Optimizer_exploration)

        #Now run the optimization iterations
        pbar = tqdm(range(self.num_of_iterations))

        for itr in pbar:
            #next values of x returned from individual function
            #Values in x are dependent on types of param dict, so using a list
            x_values_list = []

            #Next promising surrogate values for each objective: used to select functions
            s_values_array = np.empty((0,1), float)

            #keeping track of objective indices: Maps the x values to its objective function
            x_obj_indices = []

            #Next promising surrogate values for each objective in list form
            s_values_list = []

            #sample individual domains and evaluate surrogate functions.
            #we get the next promising samples along with the surrogate function values
            max_val_y = self.get_max_y_value(Y_dict_array)

            #In GPs this value is used to scale the Y values
            max_val_y_scaled = max_val_y
            #In this setting the default GP performs reasonably well
            if max_val_y_scaled<1.0:
                max_val_y_scaled = 1.0

            for j in range(len(ds)):
                if self.debug:
                    #fixing random seeds to reproduce the results for debugging
                    np.random.seed(self.seed)
                    random.seed(self.seed)

                #X_domain_np = ds[j].sample_gp_space()

                if self.first_itr[j]:
                #if j==self.last_used or self.first_itr[j]:
                    # Domain Space
                    domain_list = ds[j].get_domain()
                    self.store_domain[j] = ds[j].convert_GP_space(domain_list)
                    #self.store_domain[j] = ds[j].sample_gp_space()
                    self.first_itr[j] = True
                    #print('doing domain sampling')


                X_domain_np = self.store_domain[j]

                #next batch of x for this objective along with its surrogate value
                X_next_batch, surr_value, surr_value_ext, u_values = Optimizer_list[j].get_next_batch_MetaTuner(X_dict_array[j],Y_dict_array[j]/max_val_y_scaled,X_domain_np, self.obj_batch_size, Optimizer_exploration[j], Optimizer_iteration[j], classifier_index=j, last_used_index = j)#self.last_used

                #used for displaying and debugging
                #s_values_list.append([int(surr_value_ext[0]*100.0),int(surr_value[0]*100.0)])
                s_values_list.append([int(surr_value_ext[0]*100.0)])

                #this is used to sort the objective values
                s_value = np.array(surr_value_ext) #modified sandeep

                #s_value = np.array(surr_value_ext)
                s_value = s_value.reshape(-1, s_value.shape[0])


                s_values_array = np.append(s_values_array, s_value)
                x_values_list = x_values_list + X_next_batch

                #keep track of objective function for corresponding surrogate and x values
                for k in range(self.obj_batch_size):
                    x_obj_indices.append(j)

            #sort the surrogate values in descending order, to select the best value from them
            v_sorting_index = np.argsort(-s_values_array, axis=0)

            #now select the self.batch_size values from x_values_list based on v_sorting_index
            v_sorting_index = v_sorting_index[:self.batch_size]


            #select randomly with exploration rate else select the max surrogate

            prob_selection = random.random()

            random_selected = False

            #do the random evaluation of the functions
            if prob_selection < self.exploration_rate:
                selected_obj = random.randint(0,(len(ds)-1)) #0 and len(ds included)

                if self.exploration_rate>self.exploration_rate_min:
                    self.exploration_rate = self.exploration_rate*self.decay_rate

                else:
                    self.exploration_rate = self.exploration_rate_min

                random_selected = True

            else:
                selected_obj = v_sorting_index[0]

            #keep track of objective indices selected in current iteration
            loc_indices = []
            curr_x_next_np = x_values_list[selected_obj]

            #convert this into the parameter space for scheduling
            #see the function index for this x value
            index = selected_obj

            # updating last used classifier
            self.last_used  = index

            #keep track of local indices
            loc_indices.append(index)

            #keep track of indices in a global datastr for visualization of function selection
            self.objectives_evaluated.append(index)

            curr_x_next = ds[index].convert_PS_space(curr_x_next_np)

            #run the next curr_x_next value for the objective function
            y_list = self.objective_list[index](curr_x_next)


            self.objective_values_list += y_list

            results['params_tried'] = results['params_tried'] + curr_x_next
            results['objective_values'] = results['objective_values'] + y_list
            results['objective_fid'] = results['objective_fid'] + [index]

            curr_y_array = np.array(y_list).reshape(len(y_list), 1)
            #append the curr_x_next_np, curr_x_next, y_list to appropriate datastructures for book keeping

            X_dict_array[index] = np.vstack((X_dict_array[index], curr_x_next_np))
            Y_dict_array[index] = np.vstack((Y_dict_array[index], curr_y_array))

            Y_dict_array_max[index] = np.vstack((Y_dict_array_max[index], np.max(Y_dict_array[index])))

            #scale the exploration of objectives that are not selected for others make their exploration to 1
            for i in range(len(self.objective_list)):
                if i in loc_indices:
                    Optimizer_exploration[i] = 1.0
                    Optimizer_iteration[i] += 1.0
                else:
                    Optimizer_exploration[i] = Optimizer_exploration[i] + 1.1*self.exploration_rate


            pbar.set_description(": Best score: %s" % max_val_y)

            #print(itr, s_values_list, Optimizer_iteration, Optimizer_exploration, max_val_y)#, Y_dict_array[selected_obj])

            #print(itr, s_values_list, random_selected, self.exploration_rate, prob_selection, Optimizer_iteration , max_val_y)

            #print(itr, s_values_list, random_selected, Optimizer_iteration , int(self.exploration_rate*100.0), max_val_y)

            self.X_dict_array = X_dict_array
            self.Y_dict_array = Y_dict_array
            self.Y_dict_array_max = Y_dict_array_max
            self.ds = ds

        self.X_dict_array = X_dict_array
        self.Y_dict_array = Y_dict_array
        self.Y_dict_array_max = Y_dict_array_max
        self.ds = ds

        index_best = np.argmax(results['objective_values'])
        results['best_objective'] = results['objective_values'][index_best]
        results['best_params'] = results['params_tried'][index_best]
        results['best_objective_fid'] = results['objective_fid'][index_best]

        return results
