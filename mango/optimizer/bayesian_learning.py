import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import math


from .base_predictor import BasePredictor

"""
Bayesian Learning optimizer
- We will use the Gaussian Prior by default with the Matern Kernel
"""

class BayesianLearning(BasePredictor):


    def __init__(self):
        #initialzing some of the default values
        # The default surrogate function is gaussian_process with matern kernel
        self.surrogate = GaussianProcessRegressor(kernel=Matern(nu=2.5),n_restarts_optimizer=5,random_state =1 ,normalize_y=True)

        # keep track of the iteration counts
        self.iteration_count = 0

        # The size of the exploration domain, default to 1000
        self.domain_size = 1000

    """
    This is based on the upper confidence bound algorithm used for the aquision function
    Input: X is the values from which we need to select the best value using the aquision function.
    return: The value which is best bases on the UCB and also the mean of this value.
    """
    def Upper_Confidence_Bound(self,X):
        ''' Compute the upper confidence bound as per UCL paper
        algorithm 2 GP-BUCB: C used here is C1 value which empirically works well'''
        mu, sigma = self.surrogate.predict(X, return_std=True)
        mu = mu.reshape(mu.shape[0],1)

        sigma = sigma.reshape(sigma.shape[0],1)

        tolerance = 1e-6

        sigma_inv_sq = 1.0/(tolerance+(sigma*sigma))#tolerance is used to avoid the divide by zero error

        C = 8/(np.log(1+sigma_inv_sq))

        alpha_inter = self.domain_size*(self.iteration_count)*(self.iteration_count)*math.pi*math.pi/(6*0.1)

        if alpha_inter ==0:
            print('Error: alpha_inter is zero in Upper_Confidence_Bound')

        alpha = 2*math.log(alpha_inter) # We have set delta = 0.1
        alpha = math.sqrt(alpha)

        beta = np.exp(2*C)*alpha
        beta = np.sqrt(beta)
        Value = mu + (beta)*sigma
        x_index = np.argmax(Value)
        mu_value = mu[x_index]

        return X[x_index],mu_value

    """
    Check if the returned index value is already present in X_Sample
    """
    def Upper_Confidence_Bound_Remove_Duplicates(self,X,X_Sample):
        mu, sigma = self.surrogate.predict(X, return_std=True)
        mu = mu.reshape(mu.shape[0],1)

        sigma = sigma.reshape(sigma.shape[0],1)

        tolerance = 1e-6

        sigma_inv_sq = 1.0/(tolerance+(sigma*sigma))#tolerance is used to avoid the divide by zero error

        C = 8/(np.log(1+sigma_inv_sq))

        alpha_inter = self.domain_size*(self.iteration_count)*(self.iteration_count)*math.pi*math.pi/(6*0.1)

        if alpha_inter ==0:
            print('Error: alpha_inter is zero in Upper_Confidence_Bound')

        alpha = 2*math.log(alpha_inter) # We have set delta = 0.1
        alpha = math.sqrt(alpha)

        beta = np.exp(2*C)*alpha
        beta = np.sqrt(beta)
        Value = mu + (beta)*sigma

        return self.remove_duplicates(X,X_Sample,mu, Value)


    def remove_duplicates(self,X,X_Sample,mu, Value):
        #print('*'*200)
        v_sorting_index = np.argsort(-Value,axis=0)
        index = 0
        #go through all the values in X_Sample and check if anyvalue is close
        #to the optimal x value, if yes, don't consider this optimal x value

        while index<v_sorting_index.shape[0]:
            x_optimal = X[v_sorting_index[index]]

            #check if x_optimal is in X_Sample
            check_closeness = self.closeness(x_optimal,X_Sample)

            if check_closeness==False: #No close element to x_optimal in X_Sample
                break

                #we will look for next optimal value to try
            else:
                index = index+1

        #If entire domain is same to the already selected samples, we will just pick the best by value then
        if(index == v_sorting_index.shape[0]):
            index = 0

        return X[v_sorting_index[index]], mu[v_sorting_index[index]]


    def closeness(self,x_optimal,X_Sample):
        #check if x_optimal is close to X_Sample
        tolerance = 1e-3

        for i in range(X_Sample.shape[0]):
            diff = np.sum(np.absolute(X_Sample[i] - x_optimal))
            if(diff<tolerance):
                #print('Removed Duplicate')
                return True

        return False


    """
    This is the main function which returns the next batch to try along with the mean values for this batch
    """

    def get_next_batch(self, X,Y,X_tries,batch_size = 3):
        #print('In get_next_batch')

        X_temp = X
        Y_temp = Y

        batch=[]

        for i in range(batch_size):
            self.iteration_count = self.iteration_count + 1
            self.surrogate.fit(X_temp, Y_temp)

            X_next,u_value = self.Upper_Confidence_Bound_Remove_Duplicates(X_tries,X_temp)

            u_value = u_value.reshape(-1,1)
            Y_temp = np.vstack((Y_temp, u_value))
            X_temp = np.vstack((X_temp, X_next))

            batch.append([X_next])

        batch = np.array(batch)

        batch = batch.reshape(-1,X.shape[1])
        return batch

    """
    Get the predictions from the surrogate function
    along with the variance
    """
    def predict(self,X):
        pred_y, sigma = self.surrogate.predict(X, return_std=True)
        return pred_y,sigma


    """
    fit the optimizer on the X and Y values
    """
    def fit(self,X,Y):
        self.surrogate.fit(X, Y)
