### Publish to PyPi
If not done already, add PyPi API token for authentication 
```shell
poetry config pypi-token.pypi <api_token>
```
```shell
> poetry version [major, minor, patch]
> poetry build
> poetry publish
> git commit -am "chore(release): vx.y.z"
> git tag -a vx.y.z -m "Release vx.y.z"
> git push origin main --tags
```

### Local development: virtualenv and tests

Create a virtual environment in the project root:
```shell
python3 -m venv venv
```

Activate it from **fish**:
```shell
source ./venv/bin/activate.fish
```

Install development dependencies and run the test suite:
```shell
pip install --upgrade pip
pip install poetry
poetry install
pytest
```

## Design notes

### 2024 Aug 16: Scaling parameters in GP space
Added an option `scale_params` to scale the parameters in GP space using min max scaler. Using 10 repeated `test_six_hump` test with param `y` scaled by a factor of 1000 we get the following results:

Base case (y param not scaled in inputs or GP space): 0 out of 10 tests failed
Y param not scaled, GP space scaled:                  1 out of 10 tests failed
Without scaling GP space:                             0 out of 10 tests failed 
With scaling GP space:                                2 out of 10 tests failed

Scaling is not clearly useful so not exposing it as an option.
