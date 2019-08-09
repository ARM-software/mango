# Mango: A parallel black-box optimization library

**Mango** is a python library for parallel optimization over complex search spaces. Current usage of Mango is intended to find the optimal hyperparameters for machine learning algorithms.

# Mango Setup
```
Clone the Mango repository, and from the Mango directory
$ pip3 install -r requirements.txt
```

- Mango requires scikit-learn and is develped for python 3.
- Other packages installed are required to use Mango to optimize xgboost ML algorithms and fbprophet algorithm.

# Mango Usage
Mango is very easy to use. 
The examples are available in the directory *examples*.
General usage is as below:

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

Output

```
best hyper parameters: {'a': 999}
best objective: 6.906754778648554
```

# More Details
Details about specifying parameter/variable domain space, user objective function and internals of Mango will be added.
Please stay tuner. For any questions feel free to reach out to Sandeep Singh Sandha (sandeepsingh.sandha@arm.com)