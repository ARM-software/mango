"""
This is the file running on all the workers.
They will run the classifier with the desired hyper parameters
And return back the results.
"""

from __future__ import absolute_import, unicode_literals
from __future__ import absolute_import, unicode_literals
from celery import Celery

#whether sklearn_xgboost models should be enables on Celery
include_sklearn_xgboost = True

#whether prophet model should be enabled on Celery
include_prophet = True


app = Celery('AutoTuner',
             broker='amqp://',
             backend='rpc://')

# Optional configuration
app.conf.update(
    result_expires=3600,
    broker_heartbeat = 0
)



if include_sklearn_xgboost:
    """
    All the Classifier Functions from sklearn
    """
    import sklearn

    """
    All the Classifier Functions from xgboost
    """
    import xgboost

    from sklearn.model_selection import cross_val_score
    import numpy as np
    from importlib import import_module

    #Global variables to identify dataset is loaded by the worker
    X = None
    Y = None
    worker_dataset_name = None

    #global variables to identify classifier loaded by the worker
    clf_fxn = None
    worker_clf_name = None

    num_counter = 0

    # load the dataset for the classifier
    def get_data_loader(dataset_name):
        global worker_dataset_name
        module = import_module('sklearn.datasets')
        data_loader = getattr(module,dataset_name)
        worker_dataset_name = dataset_name
        return data_loader

    # load the classifier as passed to the worker
    def get_clf(clf_name):
        global worker_clf_name
        worker_clf_name = clf_name

        for module in sklearn.__all__:
            try:
                module = import_module(f'sklearn.{module}')
                try:
                    for clf in module.__all__:
                        if clf ==clf_name:
                            clf_function = getattr(module,clf_name)
                            return clf_function
                except:
                    pass
            except:
                pass

        for module in xgboost.__all__:
            try:
                if module ==clf_name:
                    module = import_module(f'xgboost')
                    clf_function = getattr(module,clf_name)
                    return clf_function
            except Exception as e:
                print(e)

    @app.task
    def run_clf_celery(clf_name,
                dataset_name,
                hyper_par=None):
        global X, Y, clf_fxn, worker_clf_name, worker_dataset_name, num_counter

        num_counter = num_counter+1
        #print('Worked is called:',num_counter)

        #load dataset if not done already
        if worker_dataset_name!=dataset_name:
            data_loader = get_data_loader(dataset_name)
            X,Y= data_loader(return_X_y=True)

        #load classifier if not done already
        if worker_clf_name!=clf_name:
            clf_fxn = get_clf(clf_name)

        #Assign the hyper parameters to the classifier
        if hyper_par!=None:
            clf = clf_fxn(**hyper_par)
        else:
            clf = clf_fxn()

        accuracy = cross_val_score(clf, X, Y, cv=3, scoring='accuracy').mean()
        #print('accuracy is:',accuracy)
        return accuracy


# If include_prophet is set to true
if include_prophet:

    """
    Enabling the functionality of running prophet on PJME
    """

    import numpy as np
    from prophet import Prophet
    from xgboosttree import Xgboosttree
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error



    model = Xgboosttree()

    #X_train, y_train = model.load_train_dataset("data/PJME/train_data")
    X_train, y_train = model.load_train_dataset("../classifiers/data/PJME/train_data")
    X_validate, y_validate = model.load_train_dataset("../classifiers/data/PJME/validate_data")

    @app.task
    def run_prophet(hyper_par):
        global X_train, y_train,X_validate,y_validate
        clf = Prophet(**hyper_par)
        clf.fit(X_train, y_train.ravel())
        y_pred = clf.predict(X_validate)
        mse = mean_squared_error(y_validate, y_pred)
        mse = mse/10e5
        result =  (-1.0) * mse

        return result
