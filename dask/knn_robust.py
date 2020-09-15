import json
import logging
from functools import partial

from dask.distributed import Client
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from mango import Tuner

_logger = logging.getLogger(__name__)


def dask_scheduler(client, func, params_batch, timeout=None):
    futures = [client.submit(func, params) for params in params_batch]
    results = []
    params_evaluated = []
    for params, future in zip(params_batch, futures):
        try:
            result = future.result(timeout=timeout)
            params_evaluated.append(params)
            results.append(result)
            _logger.debug(json.dumps(dict(params=params, cv_score=result)))

        except Exception:
            _logger.exception(f'Exception raised for {params}, ignoring the value.')
            continue

    return params_evaluated, results


if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # search space for KNN classifier's hyperparameters
    param_space = dict(n_neighbors=range(1, 50),
                       algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'])

    # scoring function to calculate the cross val score for a given parameters
    def cv_scorer(params):
        X, y = datasets.load_breast_cancer(return_X_y=True)
        clf = KNeighborsClassifier(**params)
        score = cross_val_score(clf, X, y, scoring='accuracy').mean()
        return score

    client = Client()
    objective_function = partial(dask_scheduler, client, cv_scorer, timeout=600)

    config = dict(num_iteration=10, batch_size=4)
    tuner = Tuner(param_space, objective_function, conf_dict=config)
    results = tuner.maximize()

    print('best parameters:', results['best_params'])
    print('best accuracy:', results['best_objective'])

    client.close()
