from dask.distributed import Client
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from mango import Tuner


def cv_scorer(params):
    '''
    Returns the cross val score for a given parameter
    '''
    X, y = datasets.load_breast_cancer(return_X_y=True)
    clf = KNeighborsClassifier(**params)
    score = cross_val_score(clf, X, y, scoring='accuracy').mean()
    return score


def dask_scheduler(params_batch):
    futures = [client.submit(cv_scorer, params) for params in params_batch]
    return [future.result() for future in futures]


if __name__ == "__main__":
    # search space for KNN classifier's hyperparameters
    param_space = dict(n_neighbors=range(1, 50),
                       algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'])

    global client
    client = Client()

    config = dict(num_iteration=5, batch_size=2)
    tuner = Tuner(param_space, dask_scheduler, conf_dict=config)
    results = tuner.maximize()

    print('best parameters:', results['best_params'])
    print('best accuracy:', results['best_objective'])

    client.close()