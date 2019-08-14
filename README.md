# Mango: A parallel black-box optimization library

**Mango** is a python library for parallel optimization over complex search spaces. Currently Mango is intended to find the optimal hyperparameters for machine learning algorithms.
Mango internally uses parallel implementation of multi-armed bandit bayesian optimizer based on gaussian process. Some of the salient features of Mango are:
- Ability to easily define complex search spaces which are compatible with the scikit-learn random search and gridsearch functions.
- Internally uses state of art optimizer which allows to sample a batch of values in parallel for evaluation.
- Objective function can be arbitrary complex which can be scheduled on local, cluster  or cloud infrastructure.
- The ease of usage was kept in mind with ability to plugin new distributions for search space and new optimizer algorithms.

## Index
1. [ Mango Setup ](#setup)
2. [ Mango Simple Example ](#simpleexample)
3. [ Tune Hyperparameters of KNeighborsClassifier ](#knnexample)
4. [ Tune Hyperparameters of Facebook Prophet ](https://gitlab.com/arm-research/isr/mango/blob/master/mango/examples/Prophet_Classifier.ipynb)
5. [ Tune Hyperparameters of xgboost XGBRegressor ](https://gitlab.com/arm-research/isr/mango/blob/master/mango/examples/Xgboost_Example.ipynb)
6. [ Parallel Scheduling using Celery](#Celery)
7. [ More Examples](https://gitlab.com/arm-research/isr/mango/tree/master/mango/examples)
8. [ Contact & Questions ](#contactDetails)

<a name="setup"></a>
# Mango Setup
```
Clone the Mango repository, and from the Mango directory.
$ pip3 install -r requirements.txt
$ python setup.py install
```

<!--
- Mango requires scikit-learn and is develped for python 3, some other packages are installed which required to optimize xgboost classifiers and fbprophet.
!-->

Testing the installation.
```
$ cd PATH/mango/mango/test
$ pytest
```


<a name="simpleexample"></a>
# Mango Simple Example
Mango is very easy to use. The given example finds optimal value of the identity function whose input is a single variable between 1 and 1000.
More examples are available in the directory *mango/examples*.

```python
from mango.tuner import Tuner

param_dict = {"a": range(1,1000)} # Search space of variables
             
def objectiveFunction(args_list): # Identity Objective Function
    a = args_list[0]['a']
    return [a]

tuner_identity = Tuner(param_dict, objectiveFunction) # Initialize Tuner

results = tuner_identity.run() # Run Tuner
print('Optimal value of a:',results['best_hyper_parameter'],' and objective:',results['best_objective'])
```

Sample output of above example.

```
Optimal value of a: {'a': 999}  and objective: 999
```
More details about this simple example are available [here.](https://gitlab.com/arm-research/isr/mango/blob/master/mango/examples/Getting_Started.ipynb)

<a name="knnexample"></a>
# Mango Example to Tune Hyperparameters of KNeighborsClassifier

```python
from mango.tuner import Tuner

from scipy.stats import uniform

# n_neighbors can vary between 1 and 50, with different choices of algorithm
param_dict = {"n_neighbors": range(1, 50),
              'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
             }
             

# Objective function for KNN
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

X, Y = datasets.load_breast_cancer(return_X_y=True)

def objectiveKNN(args_list): # arg_list is list of hyperpar values
    global X,Y # Data is loaded only once
    evaluations = []
    for hyper_par in args_list:
        clf = KNeighborsClassifier(**hyper_par)
        accuracy  = cross_val_score(clf, X, Y, scoring='accuracy').mean()
        evaluations.append(accuracy)
    return evaluations

tuner_knn = Tuner(param_dict, objectiveKNN)
results = tuner_knn.run()

print('best hyper parameters:',results['best_hyper_parameter'])
print('best Accuracy:',results['best_objective'])
```
Sample output of above example. Note output may be different for your program.

```
best hyper parameters: {'algorithm': 'auto', 'n_neighbors': 11}
best Accuracy: 0.931486122714193
```

<a name="Celery"></a>
# Mango Example to Schedule Objective Function on Celery
User defined objective function can be scheduled on local, cluster  or cloud infrastructure. We have provided sample
examples where celery workers are used to evaluate the objective function in parallel.

- [KNN example using celery workers](https://gitlab.com/arm-research/isr/mango/blob/master/mango/examples/KNN_Celery.ipynb)
- [Prophet example using celery workers](https://gitlab.com/arm-research/isr/mango/blob/master/mango/examples/Prophet_Celery.ipynb)

<a name="contactDetails"></a>
# More Details
Details about specifying parameter/variable domain space, user objective function and internals of Mango will be added.
Please stay tuned. For any questions feel free to reach out to Sandeep Singh Sandha (iotresearch@arm.com)