import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import math
import scipy

# used for clustering implementation
from sklearn.cluster import KMeans

from .base_predictor import BasePredictor

"""
Bayesian Learning optimizer
- We will use the Gaussian Prior by default with the Matern Kernel
"""


def cmaes_optimization(obj_func, initial_theta, bounds):
    import cma
    initial_sigma = 0.3
    # assuming bounds are same for all dimensions
    cmaes_bounds = [[b[0] for b in bounds], [b[1] for b in bounds]]
    opts = cma.CMAOptions()
    opts.set("bounds", cmaes_bounds)
    opts.set("verbose", -1)
    opts.set("verb_log", 0)
    opts.set("verb_disp", 0)
    opts.set('tolfun', 1e-2)
    es = cma.CMAEvolutionStrategy(initial_theta, initial_sigma, opts)
    es.optimize(obj_func)
    return es.result.xbest, es.result.fbest


class BayesianLearning(BasePredictor):

    def __init__(self, surrogate=None, n_features=None):
        # initialzing some of the default values
        # The default surrogate function is gaussian_process with matern kernel
        if surrogate is None:
            if n_features is not None:
                # anisotropic kernel
                length_scale = [2.] * n_features
            else:
                length_scale = 2.

            self.surrogate = GaussianProcessRegressor(kernel=Matern(nu=2.5,
                                                                    length_scale=length_scale,
                                                                    length_scale_bounds=(0.1, 1024)),
                                                      n_restarts_optimizer=3,
                                                      # random_state=1,
                                                      # optimizer=None,
                                                      normalize_y=False)
        else:
            self.surrogate = surrogate

        # keep track of the iteration counts
        self.iteration_count = 0

        # The size of the exploration domain, default to 1000
        self.domain_size = 1000

    """
    This is based on the upper confidence bound algorithm used for the aquision function
    Input: X is the values from which we need to select the best value using the aquision function.
    return: The value which is best bases on the UCB and also the mean of this value.
    """

    def Upper_Confidence_Bound(self, X):
        ''' Compute the upper confidence bound as per UCL paper
        algorithm 2 GP-BUCB: C used here is C1 value which empirically works well'''
        mu, sigma = self.surrogate.predict(X, return_std=True)
        mu = mu.reshape(mu.shape[0], 1)

        sigma = sigma.reshape(sigma.shape[0], 1)

        tolerance = 1e-6

        sigma_inv_sq = 1.0 / (tolerance + (sigma * sigma))  # tolerance is used to avoid the divide by zero error

        C = 8 / (np.log(1 + sigma_inv_sq))

        alpha_inter = self.domain_size * (self.iteration_count) * (self.iteration_count) * math.pi * math.pi / (6 * 0.1)

        if alpha_inter == 0:
            print('Error: alpha_inter is zero in Upper_Confidence_Bound')

        alpha = 2 * math.log(alpha_inter)  # We have set delta = 0.1
        alpha = math.sqrt(alpha)

        beta = np.exp(2 * C) * alpha
        beta = np.sqrt(beta)
        Value = mu + (beta) * sigma
        x_index = np.argmax(Value)
        mu_value = mu[x_index]

        return X[x_index], mu_value

    """
    Check if the returned index value is already present in X_Sample
    """

    def Upper_Confidence_Bound_Remove_Duplicates(self, X, X_Sample, batch_size):
        mu, sigma = self.surrogate.predict(X, return_std=True)
        mu = mu.reshape(mu.shape[0], 1)

        sigma = sigma.reshape(sigma.shape[0], 1)

        tolerance = 1e-6

        sigma_inv_sq = 1.0 / (tolerance + (sigma * sigma))  # tolerance is used to avoid the divide by zero error

        C = 8 / (np.log(1 + sigma_inv_sq))

        alpha_inter = self.domain_size * (self.iteration_count) * (self.iteration_count) * math.pi * math.pi / (6 * 0.1)

        if alpha_inter == 0:
            print('Error: alpha_inter is zero in Upper_Confidence_Bound')

        alpha = 2 * math.log(alpha_inter)  # We have set delta = 0.1
        alpha = math.sqrt(alpha)

        beta = np.exp(2 * C) * alpha
        beta = np.sqrt(beta)

        if batch_size == 1:
            exploration_factor = alpha
        else:
            exploration_factor = beta

        Value = mu + exploration_factor * sigma

        return self.remove_duplicates(X, X_Sample, mu, Value)

    """
    Returns the acqutition function
    """

    def Get_Upper_Confidence_Bound(self, X):
        mu, sigma = self.surrogate.predict(X, return_std=True)
        mu = mu.reshape(mu.shape[0], 1)

        sigma = sigma.reshape(sigma.shape[0], 1)
        alpha_inter = self.domain_size * (self.iteration_count) * (self.iteration_count) * math.pi * math.pi / (6 * 0.1)

        if alpha_inter == 0:
            print('Error: alpha_inter is zero in Upper_Confidence_Bound')

        alpha = 2 * math.log(alpha_inter)  # We have set delta = 0.1
        alpha = math.sqrt(alpha)

        Value = mu + (alpha) * sigma

        return Value

    """
    Returns the most optmal x along with mean value from the domain of x and making sure it is not a Duplicate (depending on closeness)
    used in batch setting: As mean is also returned
    """

    def remove_duplicates(self, X, X_Sample, mu, Value):
        # print('*'*200)
        v_sorting_index = np.argsort(-Value, axis=0)
        index = 0
        # go through all the values in X_Sample and check if anyvalue is close
        # to the optimal x value, if yes, don't consider this optimal x value

        while index < v_sorting_index.shape[0]:
            x_optimal = X[v_sorting_index[index]]

            # check if x_optimal is in X_Sample
            check_closeness = self.closeness(x_optimal, X_Sample)

            if check_closeness == False:  # No close element to x_optimal in X_Sample
                break

                # we will look for next optimal value to try
            else:
                index = index + 1

        # If entire domain is same to the already selected samples, we will just pick the best by value then
        if (index == v_sorting_index.shape[0]):
            index = 0

        return X[v_sorting_index[index]], mu[v_sorting_index[index]]

    """
    Returns the most optmal x only from the domain of x and making sure it is not a Duplicate (depending on closeness)
    Intended for usage in serial and clustering setting: As no mean is also returned, and no hullicination is considered
    """

    def remove_duplicates_serial(self, X, X_Sample, Value):
        # print('*'*200)
        v_sorting_index = np.argsort(-Value, axis=0)
        index = 0
        # go through all the values in X_Sample and check if anyvalue is close
        # to the optimal x value, if yes, don't consider this optimal x value

        while index < v_sorting_index.shape[0]:
            x_optimal = X[v_sorting_index[index]]

            # check if x_optimal is in X_Sample
            check_closeness = self.closeness(x_optimal, X_Sample)

            if check_closeness == False:  # No close element to x_optimal in X_Sample
                break

                # we will look for next optimal value to try
            else:
                index = index + 1

        # If entire domain is same to the already selected samples, we will just pick the best by value then
        if (index == v_sorting_index.shape[0]):
            index = 0

        return X[v_sorting_index[index]]

    def closeness(self, x_optimal, X_Sample):
        # check if x_optimal is close to X_Sample
        tolerance = 1e-3

        for i in range(X_Sample.shape[0]):
            diff = np.sum(np.absolute(X_Sample[i] - x_optimal))
            if (diff < tolerance):
                # print('Removed Duplicate')
                return True

        return False

    """
    This is the main function which returns the next batch to try along with the mean values for this batch
    """

    def get_next_batch(self, X, Y, X_tries, batch_size=3):
        # print('In get_next_batch')

        X_temp = X
        Y_temp = Y

        batch = []

        for i in range(batch_size):
            self.iteration_count = self.iteration_count + 1
            self.surrogate.fit(X_temp, Y_temp)

            X_next, u_value = self.Upper_Confidence_Bound_Remove_Duplicates(X_tries, X_temp, batch_size)
            u_value = u_value.reshape(-1, 1)
            Y_temp = np.vstack((Y_temp, u_value))
            X_temp = np.vstack((X_temp, X_next))

            batch.append([X_next])

        batch = np.array(batch)

        batch = batch.reshape(-1, X.shape[1])
        return batch

    """
    Using clustering to select next batch
    """

    def get_next_batch_clustering(self, X, Y, X_tries, batch_size=3):
        # print('In get_next_batch')

        X_temp = X
        Y_temp = Y

        self.surrogate.fit(X_temp, Y_temp)
        self.iteration_count = self.iteration_count + 1

        Acquition = self.Get_Upper_Confidence_Bound(X_tries)

        if batch_size > 1:
            kmeans = KMeans(n_clusters=4, random_state=0).fit(Acquition)
            cluster_pred = kmeans.labels_.reshape(kmeans.labels_.shape[0])
            # select the best cluster in the acquition function, and now cluster in the domain space itself
            acq_cluster_max_index = np.argmax(kmeans.cluster_centers_)

            # select the points in acq_cluster_max_index
            x_best_acq_domain = []
            x_best_acq_value = []

            for i in range(X_tries.shape[0]):
                if cluster_pred[i] == acq_cluster_max_index:
                    x_best_acq_domain.append(X_tries[i])
                    x_best_acq_value.append(Acquition[i])

            x_best_acq_domain = np.array(x_best_acq_domain)
            x_best_acq_value = np.array(x_best_acq_value)

            # Do the domain space based clustering on the best points
            kmeans = KMeans(n_clusters=batch_size, random_state=0).fit(x_best_acq_domain)
            cluster_pred_domain = kmeans.labels_.reshape(kmeans.labels_.shape[0])

            # partition the space into the cluster in X and select the best X from each space
            partitioned_space = dict()
            partitioned_acq = dict()
            for i in range(batch_size):
                partitioned_space[i] = []
                partitioned_acq[i] = []

            for i in range(x_best_acq_domain.shape[0]):
                partitioned_space[cluster_pred_domain[i]].append(x_best_acq_domain[i])
                partitioned_acq[cluster_pred_domain[i]].append(x_best_acq_value[i])

            batch = []

            for i in partitioned_space:
                x_local = partitioned_space[i]
                acq_local = partitioned_acq[i]
                acq_local = np.array(acq_local)
                x_index = np.argmax(acq_local)
                x_final_selected = x_local[x_index]
                batch.append([x_final_selected])

        else:  # batch_size ==1
            batch = []
            x_index = np.argmax(Acquition)
            x_final_selected = self.remove_duplicates_serial(X_tries, X_temp, Acquition)
            # x_final_selected = X_tries[x_index]
            batch.append([x_final_selected])

        batch = np.array(batch)
        batch = batch.reshape(-1, X.shape[1])
        return batch

    """
    Get the predictions from the surrogate function
    along with the variance
    """

    def predict(self, X):
        pred_y, sigma = self.surrogate.predict(X, return_std=True)
        return pred_y, sigma

    """
    fit the optimizer on the X and Y values
    """

    def fit(self, X, Y):
        self.surrogate.fit(X, Y)
