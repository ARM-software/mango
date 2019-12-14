from mango.tuner import Tuner
from mango.scheduler import simple_local

# search space
param_space = dict(x=range(-10, 10))


# quadratic objective function
@simple_local
def objective(x):
    return x * x


if __name__ == "__main__":
    tuner = Tuner(param_space, objective)  # Initialize Tuner
    results = tuner.minimize()  # Run Tuner
    print(f'Optimal value of parameters: {results["best_params"]} and objective: {results["best_objective"]}')