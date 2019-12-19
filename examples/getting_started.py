from mango import Tuner, scheduler

# search space
param_space = dict(x=range(-10, 10))


# quadratic objective function
@scheduler.serial
def objective(x):
    return x * x


def main():
    tuner = Tuner(param_space, objective)  # Initialize Tuner
    results = tuner.minimize()  # Run Tuner
    best_params = results["best_params"]
    best_objective = results["best_objective"]
    print(f'Optimal value of parameters: {best_params} and objective: {best_objective}')
    assert abs(best_params['x'] - 0) < 1
    assert abs(best_objective - 0) < 1


if __name__ == "__main__":
    import sys
    sys.path.extend(['/Users/mohagg01/OneDrive - Arm/workspace/mango'])
    main()