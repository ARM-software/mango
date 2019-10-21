from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import pandas as pd
import joblib
import numpy as np
from xgboost import XGBRegressor

class Xgboosttree(BaseEstimator, RegressorMixin):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(
            self,
            max_depth=3,
            learning_rate=0.1,
            n_estimators=1000,
            verbosity=1,
            silent=False,
            objective='reg:linear',
            booster='gbtree',
            n_jobs=1,
            nthread=None,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            colsample_bynode=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=0.5,
            random_state=0,
            seed=None,
            missing=None,
            importance_type='gain',
    ):

        self.max_depth = max_depth
        self.learning_rate = float(learning_rate)
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.silent = silent
        self.objective = objective
        self.booster = booster
        self.n_jobs = n_jobs
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = float(base_score)
        self.random_state = random_state
        self.seed = seed
        self.missing = missing
        self.importance_type = importance_type


    def create_features(self, df):
        """
        Creates time series features from datetime index
        """
        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['second'] = df['date'].dt.second
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear

        X = df[['hour','minute','second','dayofweek', 'quarter', 'month', 'year',
                'dayofyear', 'dayofmonth', 'weekofyear']]
        return X


    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        if X.ndim > 1:
            raise ValueError("Currently only one dim X with datetime as a column is supported")
        else:
            X = X.reshape(-1, 1)
        X, y = check_X_y(X, y, accept_sparse=True)
        self.X_ = X
        self.y_ = y
        # TODO parse X into time and exogenous variables
        self.df_ = pd.DataFrame(y, index=X[:, 0])

        X_train = self.create_features(self.df_)
        self.xgb_model_ = XGBRegressor(**self.get_params())
        self.xgb_model_.fit(X_train, y)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples. For prophet the expected shape of X is
            (n_samples, 1), where the column values are the timestamps.
            However sklearn standard API is to accept X of shape (n_samples, n_features),
            where n_features > 1.
            As a workaround  within this function we only use the first column as the input to
            the fb_prophet model.
        Returns
        -------
        y_predict : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = X.reshape(-1, 1)
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        df_x = pd.DataFrame(index=X[:, 0])
        X_predict_features = self.create_features(df_x)
        df_predict = self.xgb_model_.predict(X_predict_features)
        y_predict = pd.Series(df_predict, name='yhat')
        return y_predict


    def save_model(self, f):
        """
        Save trained model to a file
        Parameters
        ----------
        f: File name with path

        Returns
        -------
        res: bool
             Returns Filename
        """
        if not self.is_fitted_:
            raise ValueError("Can't save model as not trained yet")

        with open(f, 'wb') as o:
            res = joblib.dump(self, o, compress=True)

        return res


    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "verbosity": self.verbosity,
            "silent": self.silent,
            "objective": self.objective,
            "booster": self.booster,
            "n_jobs": self.n_jobs,
            "nthread": self.nthread,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "max_delta_step": self.max_delta_step,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "colsample_bynode": self.colsample_bynode,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "base_score": self.base_score,
            "random_state": self.random_state,
            "seed": self.seed,
            "missing": self.missing,
            "importance_type": self.importance_type
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def load_train_dataset(self, f):
        """Load X_train and y_train from given csv
        """
        df = pd.read_csv(
            f,
            index_col=[0],
            parse_dates=[0]
        )
        df = df.resample('D').mean()
        X_train = df.index.values
        y_train = df.values
        return X_train, y_train

    def load_test_dataset(self, f):
        """Load X_train and y_train from given csv
        """
        df = pd.read_csv(
            f,
            index_col=[0],
            parse_dates=[0],
            infer_datetime_format=True
        )
        #df = df.resample('D').mean()
        X = df.index.values
        return X
