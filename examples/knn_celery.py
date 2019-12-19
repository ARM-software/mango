from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import celery

from mango import Tuner, scheduler

# celery app and task
app = celery.Celery('knn_celery', backend='rpc://')


@app.task
def remote_objective(**params):
    X, y = datasets.load_breast_cancer(return_X_y=True)
    clf = KNeighborsClassifier(**params)
    score = cross_val_score(clf, X, y, scoring='accuracy').mean()
    return score


@scheduler.custom(n_jobs=2)
def objective(params_batch):
    jobs = celery.group(remote_objective.s(**params) for params in params_batch)()
    return jobs.get()


# search space for KNN classifier's hyperparameters
# n_neighbors can vary between 1 and 50, with different choices of algorithm
param_space = dict(n_neighbors=range(1, 50),
                   algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'])


def main():
    tuner = Tuner(param_space, objective, {'num_iteration': 30})
    results = tuner.maximize()

    print('best parameters:', results['best_params'])
    print('best accuracy:', results['best_objective'])

    assert results['best_objective'] > 0.93


if __name__ == "__main__":
    main()

