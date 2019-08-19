"""
Testing the capabilities of Mango
- Test the domain space transformations
- Test the sampling capabilities
- Test the bayesian learning optimizer iterations
- Test the results of tuner for simple objective function
"""
import math

from mango.domain.domain_space import domain_space
from mango.optimizer.bayesian_learning import BayesianLearning
from scipy.stats import uniform
from mango.tuner import Tuner

# Simple param_dict
param_dict = {"a": uniform(0, 1), # uniform distribution
              "b": range(1,5), # Integer variable
              "c":[1,2], # Integer variable
              "d":["-1","1"] # Categorical variable
             }

# Simple objective function
def objectiveFunction(args_list):
    results = []
    for hyper_par in args_list:
        a = hyper_par['a']
        b = hyper_par['b']
        c = hyper_par['c']
        d = hyper_par['d']
        result = (a + b + c + int(d))
        results.append(result)
    return results

domain_size = 10

#test the functionality of domain space transformations
def test_domain():
    ds = domain_space(param_dict,domain_size)

    #getting the samples from the domain
    domain_list = ds.get_domain()

    #number of samples should have same size as the domain_size
    assert len(domain_list) == domain_size

    #change this into the GP domain space
    domain_np = ds.convert_GP_space(domain_list)

    #in gaussian space verifying correctness of samples sizes and structure
    assert domain_np.shape[0] == domain_size

    #test the reverse conversion
    domain_ps = ds.convert_PS_space(domain_np)

    #domain_ps and domain_list should be same
    assert type(domain_list) == type(domain_ps)
    assert len(domain_list) == len(domain_ps)

    #testing samples should be from param_dict and structure is preserved in transformations
    if len(domain_list)>0:
        l1 = domain_list[0]
        l2 = domain_ps[0]
        assert type(l1) == type(l2)
        assert len(l1.keys()) == len(l2.keys())
        #all keys should be drawn from the param_dict
        assert len(l1.keys()) == len(param_dict.keys())

        for key in l1.keys():
            assert key in param_dict.keys()

#test the functionality of the tuner
def test_tuner():
    tuner_user = Tuner(param_dict, objectiveFunction)
    results = tuner_user.run()
    #max objective is 8, and minimum is 1
    assert results['best_objective'] >1

# test on Rosenbrock's Valley
# Rosenbrock's valley (a.k.k the banana function) has a global optimimum lying inside a long, narrow parabolic valley with a flat floor
# def test_rosenbrock():
#     param_dict = {
#         'x': range(-10, 10),
#         'y': range(-10, 10),
#     }
#     a = 1
#     b = 100
#     x_opt = a
#     y_opt = a**2
#     def objfunc(args_list):
#         results = []
#         for hyper_par in args_list:
#             x = hyper_par['x']
#             y = hyper_par['y']
#             result = -(b*((y - x**2)**2) + ((a - x)**2))
#             results.append(result)
#         return results
#
#     #  a grid search over integer values would give the results
#     #  the number of iterations should be smaller
#     grid_search_size = len(param_dict['x']) * len(param_dict['y'])
#     config = {
#         'domain_size': 5000,
#         'num_iteration': 2 * grid_search_size
#     }
#     tuner = Tuner(param_dict, objfunc)
#     results = tuner.run()
#
#     print('best hyper parameters:',results['best_hyper_parameter'])
#     print('best Accuracy:',results['best_objective'])
#
#     assert abs(results['best_hyper_parameter']['x'] - x_opt) <= 5
#     assert abs(results['best_hyper_parameter']['x'] - y_opt) <= 5

def test_convex():
    param_dict = {
        'x': range(-100, 10),
        'y': range(-10, 20),
    }

    x_opt = 0
    y_opt = 0
    def objfunc(args_list):
        results = []
        for hyper_par in args_list:
            x = hyper_par['x']
            y = hyper_par['y']
            result = -(x**2 + y**2)
            results.append(result)
        return results

    #  a grid search over integer values would give the results
    #  the number of iterations should be smaller
    grid_search_size = len(param_dict['x']) * len(param_dict['y'])
    config = {
        'domain_size': 5000,
        'num_iteration': 0.1 * grid_search_size
    }
    tuner = Tuner(param_dict, objfunc)
    results = tuner.run()

    print('best hyper parameters:',results['best_hyper_parameter'])
    print('best Accuracy:',results['best_objective'])

    assert abs(results['best_hyper_parameter']['x'] - x_opt) <= 1
    assert abs(results['best_hyper_parameter']['x'] - y_opt) <= 1


def test_six_hump():
    def camel(x,y):
        x2 = math.pow(x,2)
        x4 = math.pow(x,4)
        y2 = math.pow(y,2)
        return (4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 + x*y + (-4.0 + 4.0 * y2) * y2

    param_dict = {
        'x': uniform(-3, 3),
        'y': uniform(-2, 2),
    }

    x_opt = 0.0898 # or -0;0898
    y_opt = -0.7126  # or 0.7126
    def objfunc(args_list):
        results = []
        for hyper_par in args_list:
            x = hyper_par['x']
            y = hyper_par['y']
            result = - camel(x, y)
            results.append(result)
        return results

    config = {
        'domain_size': 5000,
        'num_iteration': 100
    }
    tuner = Tuner(param_dict, objfunc)
    results = tuner.run()

    print('best hyper parameters:',results['best_hyper_parameter'])
    print('best objective:',results['best_objective'])

    assert abs(results['best_hyper_parameter']['x']) - abs(x_opt) <= 0.1
    assert abs(results['best_hyper_parameter']['x']) - abs(y_opt) <= 0.1
