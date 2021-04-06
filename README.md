# Mango: A parallel black-box optimization library

**Mango** is a python library for parallel optimization over complex search spaces. Currently, Mango is intended to find the optimal hyperparameters for machine learning algorithms.

**Check out the quick 12 seconds demo** of Mango approximating a complex decision boundary of SVM

[![AirSim Drone Demo Video](documents/demo_video.png)](https://youtu.be/hFmSdDLLUfY)


Mango internally uses a parallel implementation of a multi-armed bandit bayesian optimizer based on the gaussian process. Some of the salient features of Mango are:
- Ability to easily define complex search spaces that are compatible with the scikit-learn random search and gridsearch functions.
- Internally uses state of the art optimizer, which allows sampling a batch of values in parallel for evaluation.
- The objective function can be arbitrarily complex, which can be scheduled on local, cluster,  or cloud infrastructure.
- The ease of usage was kept in mind with the ability to plugin new distributions for search space and new optimizer algorithms.

## Index
1. [Installation](#setup)
2. [Getting started ](#getting-started)
3. [Hyperparameter tuning example ](#knnexample)
4. [Search space definitions](#DomainSpace)
5. [Scheduler](#scheduler)
6. [Optional configurations](#MangoConfigurations)
7. [Additional Features](#AdditionalFeatures)

<!--
7. [Schedule Objective Function on Celery](#Celery)
8. [Algorithms](#mangoAlgorithms)
9. [ Tune Hyperparameters of Facebook Prophet ](https://github.com/ARM-software/mango/blob/master/examples/Prophet_Classifier.ipynb)
10. [ Tune Hyperparameters of xgboost XGBRegressor ](https://github.com/ARM-software/mango/blob/master/examples/Xgboost_Example.ipynb)
11. [ Tune Hyperparameters of xgboost XGBClassifier ](https://github.com/ARM-software/mango/blob/master/examples/Xgboost_XGBClassifier.ipynb)
12. [Tune Hyperparameters of SVM](https://github.com/ARM-software/mango/blob/master/examples/SVM_Example.ipynb)
13. [ More Examples](https://github.com/ARM-software/mango/tree/master/examples)
14. [ Contact & Questions ](#contactDetails)
-->

<a name="setup"></a>
## 1. Installation
Using `pip`:
```
pip install arm-mango
```
From source:
```
$ git clone https://github.com/ARM-software/mango.git
$ cd mango
$ pip3 install .
```

<!--
- Mango requires scikit-learn and is developed for python 3, some other packages are installed which required to optimize xgboost classifiers and fbprophet.
!-->


<a name="getting-started"></a>
## 2. Getting Started
Mango is straightforward to use. Following example minimizes the quadratic function whose input is an integer between -10 and 10.

```python
from mango import scheduler, Tuner

# Search space
param_space = dict(x=range(-10,10))
             
# Quadratic objective Function
@scheduler.serial
def objective(x): 
    return x * x

# Initialize and run Tuner
tuner = Tuner(param_space, objective) 
results = tuner.minimize()

print(f'Optimal value of parameters: {results["best_params"]} and objective: {results["best_objective"]}')```
# => Optimal value of parameters: {'x': 0}  and objective: 0
```

<a name="knnexample"></a>
## 3. Hyperparameter Tuning Example

```python
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from mango import Tuner, scheduler

# search space for KNN classifier's hyperparameters
# n_neighbors can vary between 1 and 50, with different choices of algorithm
param_space = dict(n_neighbors=range(1, 50),
                   algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'])


@scheduler.serial
def objective(**params):
    X, y = datasets.load_breast_cancer(return_X_y=True)
    clf = KNeighborsClassifier(**params)
    score = cross_val_score(clf, X, y, scoring='accuracy').mean()
    return score


tuner = Tuner(param_space, objective)
results = tuner.maximize()
print('best parameters:', results['best_params'])
print('best accuracy:', results['best_objective'])
# => best parameters: {'algorithm': 'auto', 'n_neighbors': 11}
# => best accuracy: 0.931486122714193
```
Note that best parameters may be different but accuracy should be ~ 0.9315. More examples are available 
in the `examples` directory ([Facebook's Prophet](https://github.com/ARM-software/mango/blob/master/examples/Prophet_Classifier.ipynb), 
[XGBoost](https://github.com/ARM-software/mango/blob/master/examples/Xgboost_XGBClassifier.ipynb), [SVM](https://github.com/ARM-software/mango/blob/master/examples/SVM_Example.ipynb)).


<a name="DomainSpace"></a>
## 4. Search Space
The search space defines the range and distribution of input parameters to the objective function. 
Mango search space is compatible with scikit-learn's parameter space definitions used in 
[RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
 or [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). 
The search space is defined as a dictionary with keys being the parameter names (string) and values being 
list of discreet choices, range of integers or the distributions. Example of some common search spaces are:

### Integer
Following space defines `x` as an integer parameters with values in `range(-10, 11)` (11 is not included):
```python
param_space = dict(x=range(-10, 11)) #=> -10, -9, ..., 10
# you can use steps for sparse ranges
param_space = dict(x=range(0, 101, 10)) #=> 0, 10, 20, ..., 100
```
Integers are uniformly sampled from the given range and are assumed to be ordered and treated as continuous variables.

### Categorical
Discreet categories can be defined as lists. For example:
```python
# string 
param_space = dict(color=['red', 'blue', 'green'])
# float
param_space = dict(v=[0.2, 0.1, 0.3])
# mixed 
param_space = dict(max_features=['auto', 0.2, 0.3])
```
Lists are uniformly sampled and are assumed to be unordered. They are one-hot encoded internally. 

### Distributions
All the distributions supported by [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) are supported. 
In general, distributions must provide a `rvs` method for sampling.

#### Uniform distribution
Using `uniform(loc, scale)` one obtains the uniform distribution on `[loc, loc + scale]`.
```python
from scipy.stats import uniform

# uniformly distributed between -1 and 1
param_space = dict(a=uniform(-1, 2))
```

#### Log uniform distribution
We have added [loguniform](https://github.com/ARM-software/mango/blob/master/mango/domain/distribution.py) distribution by extending the `scipy.stats.distributions` constructs.
Using `loguniform(loc, scale)` one obtains the loguniform distribution on <code>[10<sup>loc</sup>, 10<sup>loc + scale</sup>]</code>.
```python
from mango.domain.distribution import loguniform

# log uniformly distributed between 10^-3 and 10^-1
param_space = dict(learning_rate=loguniform(-3, 2))
```


### Hyperparameter search space examples
Example hyperparameter search space for [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html):
```python
param_space =  dict(
    max_features=['sqrt', 'log2', .1, .3, .5, .7, .9],
    n_estimators=range(10, 1000, 50), # 10 to 1000 in steps of 50
    bootstrap=[True, False],
    max_depth=range(1, 20),
    min_samples_leaf=range(1, 10)
)
```


Example search space for [XGBoost Classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier):
```python
from scipy.stats import uniform
from mango.domain.distribution import loguniform

param_space = {
    'n_estimators': range(10, 2001, 100), # 10 to 2000 in steps of 100
    'max_depth': range(1, 15), # 1 to 14
    'reg_alpha': loguniform(-3, 6),  # 10^-3 to 10^3
    'booster': ['gbtree', 'gblinear'],
    'colsample_bylevel': uniform(0.05, 0.95), # 0.05 to 1.0
    'colsample_bytree': uniform(0.05, 0.95), # 0.05 to 1.0
    'learning_rate': loguniform(-3, 3),  # 0.001 to 1
    'reg_lambda': loguniform(-3, 6),  # 10^-3 to 10^3
    'min_child_weight': loguniform(0, 2), # 1 to 100
    'subsample': uniform(0.1, 0.89) # 0.1 to 0.99
}
```

Example search space for [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html):
```python
from scipy.stats import uniform
from mango.domain.distribution import loguniform

param_dict = {
    'kernel': ['rbf', 'sigmoid'],
    'gamma': uniform(0.1, 4), # 0.1 to 4.1
    'C': loguniform(-7, 8) # 10^-7 to 10
} 
```

<a name="scheduler"></a>
## 5. Scheduler

Mango is designed to take advantage of distributed computing. The objective function can be scheduled to 
run locally or on a cluster with parallel evaluations. Mango is designed to allow the use of any distributed 
computing framework (like Celery or Kubernetes). The `scheduler` module comes with some pre-defined 
schedulers.

### Serial scheduler
Serial scheduler runs locally with one objective function evaluation at a time
```python
from mango import scheduler

@scheduler.serial
def objective(x): 
    return x * x
```

### Parallel scheduler
Parallel scheduler runs locally and uses `joblib` to evaluate the objective functions in parallel
```python
from mango import scheduler

@scheduler.parallel(n_jobs=2)
def objective(x): 
    return x * x
```
`n_jobs` specifies the number of parallel evaluations. `n_jobs = -1` uses all the available cpu cores 
on the machine. See [simple_parallel](https://github.com/ARM-software/mango/tree/master/examples/simple_parallel.py) 
for full working example.

### Custom distributed scheduler
Users can define their own distribution strategies using `custom` scheduler. To do so, users need to define
an objective function that takes a list of parameters and returns the list of results:
```python
from mango import scheduler

@scheduler.custom(n_jobs=4)
def objective(params_batch):
    """ Template for custom distributed objective function
    Args:
        params_batch (list): Batch of parameter dictionaries to be evaluated in parallel 
    
    Returns:
        list: Values of objective function at given parameters
    """
    # evaluate the objective on a distributed framework
    ...
    return results
```

For example the following snippet uses [Celery](http://www.celeryproject.org/): 
```python
import celery
from mango import Tuner, scheduler

# connect to celery backend
app = celery.Celery('simple_celery', backend='rpc://')

# remote celery task
@app.task
def remote_objective(x):
    return x * x

@scheduler.custom(n_jobs=4)
def objective(params_batch):
    jobs = celery.group(remote_objective.s(params['x']) for params in params_batch)()
    return jobs.get()

param_space = dict(x=range(-10, 10))

tuner = Tuner(param_space, objective)
results = tuner.minimize()
```
A working example to tune hyperparameters of KNN using Celery is [here](https://github.com/ARM-software/mango/tree/master/examples/knn_celery.py).
 
<!--
<a name="ObjectiveFunction"></a>
## 5. More on Objective Function
The serial objective function has the following structure.

```python
def objective_function(params_list):
    evaluations = []
    for hyper_par in params_list:
        result =  evaluate_function_on_hyper_par
        evaluations.append(result)
    return evaluations
```
The objective function is called with the input list of hyper parameters. Each element of the list is the dictionary which is a sample drawn from the domain space of variables. Mango expects the objective function to return the list of
evaluations which has the same size as the args_list. Each value of the evaluations list is the function evaluated at hyperparameters
of params_list in the same order. A rich set of objective functions are shown in the [examples](https://github.com/ARM-software/mango/tree/master/examples). The size of the params_list is controlled by the batch_size configuration parameter of Mango. By default,
batch_size is 1. The configuration parameters of Mango are explained in the [Mango Configurations](#MangoConfigurations) section. 

The sample skeleton of the Celery based parallel objective function in Mango is as following.

```python
def objective_celery(params_list):
    process_queue = []
    for par in params_list:
        process = train_clf.delay(par)
        process_queue.append((process, par))
    evals = []
    params = []
    for process, par in process_queue:
        result = process.get()
        evals.append(result)
        params.append(par)
    return params, evals
```
-->

<a name="MangoConfigurations"></a>
## 6. Optional configurations

The default configuration parameters used by the Mango as below:
```python
{'param_dict': ...,
 'userObjective': ...,
 'domain_size': 5000,
 'initial_random': 1,
 'num_iteration': 20,
 'batch_size': 1}
```
The configuration parameters are:
- domain_size: The size which is explored in each iteration by the gaussian process. Generally, a larger size is preferred if higher dimensional functions are optimized. More on this will be added with details about the internals of bayesian optimization.
- initial_random: The number of random samples tried.
- num_iteration: The total number of iterations used by Mango to find the optimal value.
- batch_size: The size of args_list passed to the objective function for parallel evaluation. For larger batch sizes, Mango internally uses intelligent sampling to decide the optimal samples to evaluate.
- early_stopping: A callback to specify custom stopping criteria. The callback has the following signature:
   ```python
  def early_stopping(results):
      '''
          results is the same as dict returned by tuner
          keys available: params_tries, objective_values, 
              best_objective, best_params
      '''
      ...
      return True/False
  ```
Early stopping is one of Mango's important features that allow to early terminate the current parallel search based on the custom user-designed criteria, such as the total optimization time spent, current validation accuracy achieved, or improvements in the past few iterations. For usage see early stopping examples [notebook](https://github.com/ARM-software/mango/blob/master/examples/EarlyStopping.ipynb).


The default configuration parameters can be modified, as shown below. Only the parameters whose values need to adjusted can be passed as the dictionary.

```python
conf_dict = dict(num_iteration=40, domain_size=10000, initial_random=3)

tuner = Tuner(param_dict, objective, conf_dict) 
```

<a name="AdditionalFeatures"></a>
## 7. Additional Features
### Handling runtime failed evaluation
At runtime, failed evaluations are widespread in production deployments. Mango abstractions enable users to make progress even in the presence of failures by only using the correct evaluations. The syntax can return the successful evaluation, and the user can flexibly keep track of failures, for example, using timeouts. An example showing the usage of Mango in the presence of random failures is shown [here](https://github.com/ARM-software/mango/blob/master/examples/Failure_Handling.ipynb).   

<!--
<a name="Celery"></a>
## 7. Schedule Objective Function on Celery
User-defined objective function can be scheduled on local, cluster or cloud infrastructure. The objective function scheduler
is entirely independent of the Mango. This design was chosen to enable the scheduling of varied resource objective function according to developer needs. We have included examples using [Celery](http://www.celeryproject.org/). In the sample
examples, celery workers are used to evaluate the objective function in parallel. These examples assume celery is installed, and workers
are running. Default celery configurations can be modified in the [file](https://github.com/ARM-software/mango/blob/master/examples/classifiers/celery.py).

- [KNN example using celery workers](https://github.com/ARM-software/mango/blob/master/examples/KNN_Celery.ipynb)
- [Prophet example using celery workers](https://github.com/ARM-software/mango/blob/master/examples/Prophet_Celery.ipynb)

More examples will be included to show the scheduling of objective function using local threads/processes. By default examples schedule
the objective function on the local machine itself. 

<a name ="mangoAlgorithms"></a>
## 8. Algorithms
The optimization algorithms in Mango are based on widely used Bayesian optimization techniques, extended to sample a batch of configurations in parallel. Currently, Mango provides two parallel optimization algorithms that use the upper confidence bound as the acquisition function. The first algorithm uses hallucination combined with exponential rescaling of the surrogate function to select a batch. In the second algorithm, we create clusters of acquisition function in spatially distinct search spaces, and select the maximum value within each cluster to create the batch. 

<a name="contactDetails"></a>
## More Details
Details about specifying parameter/variable domain space, user objective function, and internals of Mango will be added.
Please stay tuned. 
-->

## Participate

### Paper

More technical details are available in the [Mango paper (ICASSP 2020 Conference)](https://arxiv.org/pdf/2005.11394.pdf). Please cite this as:
```
@inproceedings{sandha2020mango,
  title={Mango: A Python Library for Parallel Hyperparameter Tuning},
  author={Sandha, Sandeep Singh and Aggarwal, Mohit and Fedorov, Igor and Srivastava, Mani},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3987--3991},
  year={2020},
  organization={IEEE}
}
```
### Slides

Slides explaining Mango abstractions and design choices are available. [Mango Slides](https://github.com/ARM-software/mango/blob/master/documents/Mango_github_slides.pdf).

### Contribute

Please take a look at [open issues](https://github.com/ARM-software/mango/issues) if you are looking for areas to contribute to.

### Questions

For any questions feel free to reach out by creating an issue [here](https://github.com/ARM-software/mango/issues/new).
