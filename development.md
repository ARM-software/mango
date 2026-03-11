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

## Release v1.4.3

### Changes
- Fix progress bar optional behavior
- Close #123

### Git Commands Used
```bash


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