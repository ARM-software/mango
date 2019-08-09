from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from fbprophet import Prophet as FBProphet
import pandas as pd
import joblib
from hyperopt import hp
import numpy as np

class Prophet(BaseEstimator, RegressorMixin):
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
            changepoints=None,
            n_changepoints=25,
            changepoint_range=0.8,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            holidays=None,
            seasonality_mode='additive',
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
    ):
        self.changepoints = changepoints
        self.n_changepoints = float(n_changepoints)
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = float(seasonality_prior_scale)
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = float(changepoint_prior_scale)


    @classmethod
    def hyperparameter_space(cls):
        space = {
            "changepoint_prior_scale": hp.loguniform(
                "changepoint_prior_scale",
                np.log(0.001),
                np.log(0.5)
            ),
            "seasonality_prior_scale": hp.loguniform(
                'seasonality_prior_scale',
                np.log(1),
                np.log(100)
            )
        }
        return space


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

        #print('Sandeep: clf.fit is called:')

        # TODO parse X into time and exogenous variables
        self.df_ = pd.DataFrame({'ds': X[:, 0], 'y': y})

        self.prophet_model_ = FBProphet(**self.get_params())
        self.prophet_model_.fit(self.df_)
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
        df_x = pd.DataFrame({'ds': X[:, 0]})
        df_predict = self.prophet_model_.predict(df_x)
        y_predict = df_predict['yhat']
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
            "changepoints": self.changepoints,
            "n_changepoints": self.n_changepoints,
            "changepoint_range": self.changepoint_range,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "holidays": self.holidays,
            "seasonality_mode": self.seasonality_mode,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "holidays_prior_scale": self.holidays_prior_scale,
            "changepoint_prior_scale": self.changepoint_prior_scale
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
