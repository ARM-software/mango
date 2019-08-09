# Mango: A parallel black-box optimization library

**Mango** is a python library for parallel optimization over complex search spaces. Current usage of Mango is intended to find the optimal hyperparameters for machine learning algorithms.

# Mango Setup
```
Clone the Mango repository, and from the Mango directory
$ pip3 install -r requirements.txt
```

- Mango requires scikit-learn and is develped for python 3.
- Other packages installed are required to use Mango to optimize xgboost ML algorithms and fbprophet algorithm.

# Mango Usage to Find Optimal Value of a Function
Mango is very easy to use. 
The examples are available in the directory *examples*.
In the example below our goal is to find optimal value of the function whose input is a single variable between 1 and 1000.
The objective of the function is unknown to the Mango, but it can evaluate the function for fixed number of iterations.
The selected function is log.

```
# Import Tuner 
from tuner import Tuner

# Define the domain space of the objective function to optimize using param_dict
# param_dict is a dictionary. Examples include complex space definitions.
param_dict = {
              "a": range(1,1000),
             }
             
# Define user Objective function
# Here the function is log(a), and we want to find the Maximum value of this function
import math

def objectiveFunction(args_list):
    evaluations = []
    for hyper_par in args_list:
        a = hyper_par['a']
        evaluations.append(math.log(a))
    return evaluations


# Create Tuner object and pass required parameters
tuner_user = Tuner(param_dict, objectiveFunction)

# Run the Tuner
results = tuner_user.run()

# results contains the details of the process followed by Mango
print('best hyper parameters:',results['best_hyper_parameter'])
print('best objective:',results['best_objective'])
```

Output of Running Above Program. Note output may be different for your program.

```
best hyper parameters: {'a': 999}
best objective: 6.906754778648554
```

# Mango Usage to Tune Hyperparameters of KNeighborsClassifier

```
from tuner import Tuner

from scipy.stats import uniform

# n_neighbors can vary between 1 and 50, with different choices of algorithm
param_dict = {"n_neighbors": range(1, 50),
              'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
             }
             

# User Objective function for KNN
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


X, Y = datasets.load_breast_cancer(return_X_y=True)

def objectiveKNN(args_list):
    global X,Y
    
    results = []
    for hyper_par in args_list:
        clf = KNeighborsClassifier(**hyper_par)
        result  = cross_val_score(clf, X, Y, scoring='accuracy').mean()
        results.append(result)
    return results

tuner_user = Tuner(param_dict, objectiveKNN, conf_Dict)

results = tuner_user.run()


print('best hyper parameters:',results['best_hyper_parameter'])
print('best Accuracy:',results['best_objective'])
```
Output of Running Above Program. Note output may be different for your program.

```
best hyper parameters: {'algorithm': 'auto', 'n_neighbors': 11}
best Accuracy: 0.931486122714193
```


# More Details
Details about specifying parameter/variable domain space, user objective function and internals of Mango will be added.
Please stay tuner. For any questions feel free to reach out to Sandeep Singh Sandha (sandeepsingh.sandha@arm.com)