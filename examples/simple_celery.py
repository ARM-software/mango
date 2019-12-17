import celery

from mango import Tuner, scheduler

# connect to celery backend
app = celery.Celery('simple_celery', backend='rpc://')


# remote celery task with static parameter
@app.task
def remote_objective(x, a):
    return (x - a) * (x - a)


# search space
param_space = dict(x=range(-10, 10))


# objective function with a == 5
@scheduler.celery(n_jobs=2)
def objective(x):
    return remote_objective.delay(x, 5)


def main():
    tuner = Tuner(param_space, objective)
    results = tuner.minimize()

    print('best parameters:', results['best_params'])
    print('best accuracy:', results['best_objective'])

    assert abs(results['best_objective'] - 0) < 1


if __name__ == "__main__":
    main()

