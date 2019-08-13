"""
Testing the capabilities of Mango
- Test the domain space transformations
- Test the sampling capabilities
- Test the bayesian learning optimizer iterations
- Test the results of tuner for simple objective function
"""

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
