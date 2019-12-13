from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from mango.tuner import Tuner
from mango.scheduler import simple_local

# search space for KNN classifier's hyperparameters
# n_neighbors can vary between 1 and 50, with different choices of algorithm
param_space = dict(n_neighbors=range(1, 50),
                   algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'])


@simple_local
def objective(**params):
    X, y = datasets.load_breast_cancer(return_X_y=True)
    clf = KNeighborsClassifier(**params)
    score = cross_val_score(clf, X, y, scoring='accuracy').mean()
    return score


if __name__ == "__main__":
    tuner = Tuner(param_space, objective, {'num_iterations': 30})
    results = tuner.maximize()

    print('best parameters:', results['best_params'])
    print('best accuracy:', results['best_objective'])
