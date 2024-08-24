import math
import time
import pytest
import numpy as np
import random

from mango.domain.domain_space import DomainSpace
from mango import Tuner, scheduler
from scipy.stats import uniform, dirichlet

# Simple param_dict
param_dict = {
    "a": uniform(0, 1),  # uniform distribution
    "b": range(1, 5),  # Integer variable
    "c": [1, 2],  # Integer variable
    "d": ["-1", "1"],  # Categorical variable
}


# Simple objective function
def objectiveFunction(args_list):
    results = []
    for hyper_par in args_list:
        a = hyper_par["a"]
        b = hyper_par["b"]
        c = hyper_par["c"]
        d = hyper_par["d"]
        result = a + b + c + int(d)
        results.append(result)
    return results


# test the functionality of domain space transformations
def test_domain():
    domain_size = 10
    ds = DomainSpace(param_dict)
    ds.domain_size = domain_size
    # getting the samples from the domain
    domain_list = ds.get_domain()

    # number of samples should have same size as the domain_size
    assert len(domain_list) == domain_size

    # change this into the GP domain space
    domain_np = ds.convert_GP_space(domain_list)

    # in gaussian space verifying correctness of samples sizes and structure
    assert domain_np.shape[0] == domain_size

    # test the reverse conversion
    domain_ps = ds.convert_PS_space(domain_np)

    # domain_ps and domain_list should be same
    assert type(domain_list) == type(domain_ps)
    assert len(domain_list) == len(domain_ps)

    # testing samples should be from param_dict and structure is preserved in transformations
    if len(domain_list) > 0:
        l1 = domain_list[0]
        l2 = domain_ps[0]
        assert type(l1) == type(l2)
        assert len(l1.keys()) == len(l2.keys())
        # all keys should be drawn from the param_dict
        assert len(l1.keys()) == len(param_dict.keys())

        for key in l1.keys():
            assert key in param_dict.keys()

    ps = dict(x=range(1, 100), y=["a", "b"], z=uniform(-10, 20))
    ds = DomainSpace(ps)
    ds.domain_size = 100

    x = ds.get_domain()
    x_gp = ds.convert_GP_space(x)
    x_rebuilt = ds.convert_PS_space(x_gp)
    for x1, x2 in zip(x, x_rebuilt):
        for k in x1.keys():
            v1 = x1[k]
            v2 = x2[k]
            if isinstance(v1, np.float64):
                assert v1 == pytest.approx(v2, abs=1e-5)
            else:
                if not v1 == v2:
                    print(k)
                    print(x)
                    print(x_gp)
                    print(x_rebuilt)
                assert v1 == v2


# test the functionality of the tuner
def test_tuner():
    tuner_user = Tuner(param_dict, objectiveFunction)
    results = tuner_user.run()
    # max objective is 8, and minimum is 1
    assert results["best_objective"] > 1


# test on Rosenbrock's Valley
# Rosenbrock's valley (a.k.k the banana function) has a global optimimum lying inside a long, narrow parabolic valley with a flat floor
def test_rosenbrock():
    param_dict = {
        "x": range(-10, 10),
        "y": range(-10, 10),
    }
    a = 1
    b = 100
    x_opt = a
    y_opt = a**2

    def objfunc(args_list):
        results = []
        for hyper_par in args_list:
            x = hyper_par["x"]
            y = hyper_par["y"]
            result = -(b * ((y - x**2) ** 2) + ((a - x) ** 2))
            results.append(result)
        return results

    tuner = Tuner(
        param_dict, objfunc, conf_dict=dict(domain_size=100000, num_iteration=40)
    )
    results = tuner.run()

    print("best hyper parameters:", results["best_params"])
    print("best Accuracy:", results["best_objective"])

    assert abs(results["best_params"]["x"] - x_opt) <= 2
    assert abs(results["best_params"]["x"] - y_opt) <= 2


def test_config():
    param_dict = {
        "x": range(-10, 10),
        "y": range(-10, 10),
    }

    x_opt = 0
    y_opt = 0

    def objfunc(args_list):
        results = []
        for hyper_par in args_list:
            x = hyper_par["x"]
            y = hyper_par["y"]
            result = (x**2 + y**2) / 1e4
            results.append(result)
        return results

    def check(results, error_msg):
        assert abs(results["best_params"]["x"] - x_opt) <= 3, error_msg
        assert abs(results["best_params"]["y"] - y_opt) <= 3, error_msg

    tuner = Tuner(param_dict, objfunc, conf_dict=dict(optimizer="Random"))
    results = tuner.minimize()
    check(results, "error while minimizing random")


def test_convex():
    param_dict = {
        "x": range(-100, 10),
        "y": range(-10, 20),
    }

    x_opt = 0
    y_opt = 0

    def objfunc(args_list):
        results = []
        for hyper_par in args_list:
            x = hyper_par["x"]
            y = hyper_par["y"]
            result = (x**2 + y**2) / 1e4
            results.append(result)
        return results

    tuner = Tuner(param_dict, objfunc)
    results = tuner.minimize()

    print("best hyper parameters:", results["best_params"])
    print("best Accuracy:", results["best_objective"])

    assert abs(results["best_params"]["x"] - x_opt) <= 3
    assert abs(results["best_params"]["y"] - y_opt) <= 3


def test_initial_custom():
    param_dict = {
        "x": range(-100, 10),
        "y": range(-10, 20),
    }

    x_opt = 0
    y_opt = 0

    def objfunc(args_list):
        results = []
        for hyper_par in args_list:
            x = hyper_par["x"]
            y = hyper_par["y"]
            result = (x**2 + y**2) / 1e4
            results.append(result)
        return results

    config = dict(initial_custom=[dict(x=-100, y=20), dict(x=10, y=20)])

    tuner = Tuner(param_dict, objfunc, conf_dict=config)
    results = tuner.minimize()

    print("best hyper parameters:", results["best_params"])
    print("best Accuracy:", results["best_objective"])

    assert abs(results["best_params"]["x"] - x_opt) <= 3
    assert abs(results["best_params"]["y"] - y_opt) <= 3
    assert results["random_params"][0] == config["initial_custom"][0]
    assert results["random_params"][1] == config["initial_custom"][1]


def test_local_scheduler():
    param_space = dict(x=range(-10, 10), y=range(-10, 10))

    @scheduler.serial
    def obj(x, y):
        return x - y

    results = Tuner(param_space, obj).maximize()

    assert abs(results["best_params"]["x"] - 10) <= 3
    assert abs(results["best_params"]["y"] + 10) <= 3

    @scheduler.parallel(n_jobs=-1)
    def obj(x, y):
        return x - y

    results = Tuner(param_space, obj).maximize()

    assert abs(results["best_params"]["x"] - 10) <= 3
    assert abs(results["best_params"]["y"] + 10) <= 3

    @scheduler.parallel(n_jobs=2)
    def obj(x, y):
        return x - y

    results = Tuner(param_space, obj).maximize()

    assert abs(results["best_params"]["x"] - 10) <= 3
    assert abs(results["best_params"]["y"] + 10) <= 3


@pytest.mark.parametrize(
    "y_scale, scale_params", [(1, False), (1, True), (1000.0, False), (1000.0, True)]
)
def test_six_hump(y_scale, scale_params):

    def camel(x, y):
        y = y / y_scale
        x2 = math.pow(x, 2)
        x4 = math.pow(x, 4)
        y2 = math.pow(y, 2)
        return (4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 + x * y + (-4.0 + 4.0 * y2) * y2

    param_dict = {
        "x": uniform(-3, 3),
        "y": uniform(-2 * y_scale, 2 * y_scale),
    }

    x_opt = 0.0898  # or -0.0898
    y_opt = -0.7126 * y_scale  # or 0.7126

    def objfunc(args_list):
        results = []
        for hyper_par in args_list:
            x = hyper_par["x"]
            y = hyper_par["y"]
            result = -camel(x, y)
            results.append(result)
        return results

    tuner = Tuner(param_dict, objfunc, dict(scale_params=scale_params))
    results = tuner.run()

    print("best hyper parameters:", results["best_params"])
    print("best objective:", results["best_objective"])

    assert abs(results["best_params"]["x"]) - abs(x_opt) <= 0.1
    assert abs(results["best_params"]["y"]) - abs(y_opt) <= 0.2 * y_scale


def test_celery_scheduler():
    try:
        import celery
    except ModuleNotFoundError:
        return

    # search space
    param_space = dict(x=range(-10, 10))

    class MockTask:

        def __init__(self, x):
            self.x = x

        def objective(self):
            return self.x * self.x

        def get(self, timeout=None):
            return self.objective()

    @scheduler.celery(n_jobs=2)
    def objective(x):
        return MockTask(x)

    tuner = Tuner(param_space, objective)
    assert tuner.config.batch_size == 2

    results = tuner.minimize()

    assert abs(results["best_params"]["x"]) <= 0.1

    class MockTask:

        def __init__(self, x):
            self.x = x

        def objective(self):
            return (self.x - 5) * (self.x - 5)

        def get(self, timeout=None):
            if self.x < -8:
                raise celery.exceptions.TimeoutError("timeout")
            return self.objective()

    @scheduler.celery(n_jobs=1)
    def objective(x):
        return MockTask(x)

    tuner = Tuner(param_space, objective)
    results = tuner.minimize()

    assert abs(results["best_params"]["x"] - 5) <= 0.1


def test_custom_scheduler():
    # search space
    param_space = dict(x=range(-10, 10))

    @scheduler.custom(n_jobs=2)
    def objective(params):
        assert len(params) == 2
        return [p["x"] * p["x"] for p in params]

    tuner = Tuner(param_space, objective, dict(initial_random=2))
    results = tuner.minimize()

    assert abs(results["best_params"]["x"] - 0) <= 0.1


def test_early_stopping_simple():
    param_dict = dict(x=range(-10, 10))

    def objfunc(p_list):
        return [p["x"] ** 2 for p in p_list]

    def early_stop(results):
        if len(results["params_tried"]) >= 5:
            return True

    config = dict(num_iteration=20, initial_random=1, early_stopping=early_stop)

    tuner = Tuner(param_dict, objfunc, conf_dict=config)
    results = tuner.minimize()
    assert len(results["params_tried"]) == 5

    config = dict(
        num_iteration=20,
        initial_random=1,
        early_stopping=early_stop,
        optimizer="Random",
    )

    tuner = Tuner(param_dict, objfunc, conf_dict=config)
    results = tuner.minimize()
    assert len(results["params_tried"]) == 5


def test_early_stopping_complex():
    """testing early stop by time since last improvement"""
    param_dict = dict(x=range(-10, 10))

    def objfunc(p_list):
        time.sleep(2)
        res = [np.random.uniform()] * len(p_list)
        return res

    class Context:
        previous_best = None
        previous_best_time = None
        min_improvement_secs = 1
        objective_variation = 1

    def early_stop(results):
        context = Context

        current_best = results["best_objective"]
        current_time = time.time()
        _stop = False

        if context.previous_best is None:
            context.previous_best = current_best
            context.previous_best_time = current_time
        elif (current_best <= context.previous_best + context.objective_variation) and (
            current_time - context.previous_best_time > context.min_improvement_secs
        ):
            print(
                "no improvement in %d seconds: stopping early."
                % context.min_improvement_secs
            )
            _stop = True
        else:
            context.previous_best = current_best
            context.previous_best_time = current_time

        return _stop

    config = dict(num_iteration=20, initial_random=1, early_stopping=early_stop)

    tuner = Tuner(param_dict, objfunc, conf_dict=config)
    results = tuner.minimize()
    assert len(results["params_tried"]) == 3


def test_constrainted_opt():
    param_dict = {
        "a": uniform(0, 1),  # uniform distribution
        "b": range(1, 5),  # Integer variable
        "c": [1, 2, 3],  # Integer variable
        "d": ["-1", "1"],  # Categorical variable
    }

    def constraint(samples):
        is_valid = []
        for sample in samples:
            if sample["a"] < 0.5:
                v = sample["b"] >= 3
            else:
                v = sample["b"] < 3
            is_valid.append(v)

        return is_valid

    def objectiveFunction(args_list):
        results = []
        for hyper_par in args_list:
            a = hyper_par["a"]
            b = hyper_par["b"]
            c = hyper_par["c"]
            d = hyper_par["d"]
            result = a + b + c + int(d)
            results.append(result)
        return results

    config = dict(num_iteration=20, constraint=constraint)
    tuner = Tuner(param_dict, objectiveFunction, config)

    results = tuner.maximize()

    assert results["best_objective"] > 8.4
    assert results["best_params"]["b"] == 4
    assert constraint([results["best_params"]])


def test_multivar():
    def objfun(params):
        res = []
        for param in params:
            res.append(sum([v**2 for v in param["multi"]]) - param["min"])
        return res

    param_space = {
        "multi": dirichlet([1, 1, 1]),
        "min": range(10, 100),
    }

    tuner = Tuner(param_space, objfun, conf_dict={"batch_size": 2})
    results = tuner.run()

    # for dirichlet distribution the maximum  sum of squares is at an edge
    assert results["best_params"]["min"] == pytest.approx(10)
    assert max(results["best_params"]["multi"]) == pytest.approx(1, abs=0.5)
    assert min(results["best_params"]["multi"]) == pytest.approx(0, abs=0.5)


def test_multivar_with_fails():
    @scheduler.custom(n_jobs=3)
    def objfun(batch):
        loss = []
        suc = []
        for params in batch:
            if random.choice([True, False]):
                l = params["a"] + params["b"] + sum(params["c"])
                loss.append(l)
                suc.append(params)
        return suc, loss

    param_space = {
        "a": range(1, 100),
        "b": range(1, 100),
        "c": dirichlet([1.0] * 3),
    }

    tuner = Tuner(param_space, objfun, conf_dict={"initial_random": 4})
    results = tuner.run()
    assert results["best_params"]["a"] == pytest.approx(99)
    assert results["best_params"]["b"] == pytest.approx(99)


def test_failure_handling():
    param_dict = {
        "x": uniform(-5, 10),
        "y": uniform(-5, 10),
    }

    # Randomly fail the evaluatioon
    def objfunc(args_list):
        hyper_evaluated = []
        objective_evaluated = []
        for hyper_par in args_list:
            if random.random() > 0.3:
                x = hyper_par["x"]
                y = hyper_par["y"]
                objective = -(x**2 + y**2)
                objective_evaluated.append(objective)
                hyper_evaluated.append(hyper_par)
            else:
                continue

        return hyper_evaluated, objective_evaluated

    tuner = Tuner(param_dict, objfunc)
    results = tuner.maximize()
    x = results["params_tried"]
    y = results["objective_values"]
    for params, obj in zip(x, y):
        assert obj == -(params["x"] ** 2 + params["y"] ** 2)


def test_warmup():
    def objfun(params):
        return [param["a"] + param["b"] for param in params]

    param_space = {
        "a": range(1, 100),
        "b": range(1, 100),
    }

    tuner = Tuner(param_space, objfun)
    results = tuner.run()

    x = results["params_tried"]
    y = results["objective_values"]
    xy = list(zip(x, y))

    config = {"initial_custom": xy}

    tuner = Tuner(param_space, objfun, conf_dict=config)
    results = tuner.run()

    assert all([i in results["params_tried"] for i in x])
