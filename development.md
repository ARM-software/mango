### Publish to PyPi
If not done already, add PyPi API token for authentication 
```shell
poetry config pypi-token.pypi <api_token>
```
```shell
> poetry version [major, minor, patch]
> poetry build
> poetry publish
```