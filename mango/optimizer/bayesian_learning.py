import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import math
from sklearn.cluster import KMeans

from .base_predictor import BasePredictor

"""
Bayesian Learning optimizer
- We will use the Gaussian Prior by default with the Matern Kernel
"""


class BayesianLearning(BasePredictor):

    def __init__(self, surrogate=None, alpha=None, domain_size=1000):
        # initialzing some of the default values
        # The default surrogate function is gaussian_process with matern kernel
        if surrogate is None:
            self.surrogate = GaussianProcessRegressor(kernel=Matern(nu=2.5),
                                                      n_restarts_optimizer=10,
                                                      # FIXME:check if we should be passing this
                                                      # random state
                                                      random_state=1,
                                                      normalize_y=True)
        else:
            self.surrogate = surrogate

        # keep track of the iteration counts
        self.iteration_count = 0

        # The size of the exploration domain, default to 1000
        self.domain_size = domain_size

        self.alpha = alpha

    def Upper_Confidence_Bound_Remove_Duplicates(self, X, X_Sample, batch_idx):
        """
            Check if the returned index value is already present in X_Sample
        """
        mu, sigma = self.surrogate.predict(X, return_std=True)
        mu = mu.reshape(mu.shape[0], 1)
        sigma = sigma.reshape(sigma.shape[0], 1)

        # use fixed alpha if given
        if self.alpha is not None:
            alpha = self.alpha
        else:
            alpha_inter = self.domain_size * (self.iteration_count) * (self.iteration_count) * math.pi * math.pi / (
                    6 * 0.1)

            if alpha_inter == 0:
                raise ValueError('alpha_inter is zero in Upper_Confidence_Bound')

            alpha = 2 * math.log(alpha_inter)  # We have set delta = 0.1
            alpha = math.sqrt(alpha)

        if batch_idx == 0:
            exploration_factor = alpha
        else:
            tolerance = 1e-6
            sigma_inv_sq = 1.0 / (tolerance + (sigma * sigma))  # tolerance is used to avoid the divide by zero error
            C = 8 / (np.log(1 + sigma_inv_sq))
            beta = np.exp(2 * C) * alpha
            beta = np.sqrt(beta)
            exploration_factor = beta

        Value = mu + exploration_factor * sigma

        return self.remove_duplicates(X, X_Sample, mu, Value)

    def Get_Upper_Confidence_Bound(self, X):
        """
            Returns the acqutition function
        """
        mu, sigma = self.surrogate.predict(X, return_std=True)
        mu = mu.reshape(mu.shape[0], 1)
        sigma = sigma.reshape(sigma.shape[0], 1)

        if self.alpha is not None:
            exploration_factor = self.alpha
        else:
            alpha_inter = self.domain_size * (self.iteration_count) * (self.iteration_count) * math.pi * math.pi / (
                    6 * 0.1)
            if alpha_inter == 0:
                raise ValueError('alpha_inter is zero in Upper_Confidence_Bound')
            alpha = 2 * math.log(alpha_inter)  # We have set delta = 0.1
            alpha = math.sqrt(alpha)

            exploration_factor = alpha

        Value = mu + exploration_factor * sigma

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

    def get_next_batch(self, X, Y, X_tries, batch_size):
        # print('In get_next_batch')

        X_temp = X
        Y_temp = Y

        batch = []

        for idx in range(batch_size):
            self.iteration_count = self.iteration_count + 1
            self.surrogate.fit(X_temp, Y_temp)

            X_next, u_value = self.Upper_Confidence_Bound_Remove_Duplicates(X_tries, X_temp, idx)
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

    def get_next_batch_clustering(self, X, Y, X_tries, batch_size):
        # print('In get_next_batch')

        X_temp = X
        Y_temp = Y

        self.surrogate.fit(X_temp, Y_temp)
        self.iteration_count = self.iteration_count + 1

        Acquition = self.Get_Upper_Confidence_Bound(X_tries)

        if batch_size > 1:
            gen = sorted(zip(Acquition, X_tries), key=lambda x: -x[0])
            x_best_acq_value, x_best_acq_domain = (np.array(t)[:len(Acquition) // 4]
                                                   for t in zip(*gen))

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
