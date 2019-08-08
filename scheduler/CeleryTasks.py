"""
This is the file running on all the workers.
They will run the classifier with the desired hyper parameters
And return back the results.
"""

from __future__ import absolute_import, unicode_literals
from __future__ import absolute_import, unicode_literals
from celery import Celery

app = Celery('AutoTuner',
             broker='amqp://',
             backend='rpc://')

# Optional configuration
app.conf.update(
    result_expires=3600,
    broker_heartbeat = 0
)



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
