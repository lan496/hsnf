## Documentation

Local build
```shell
sphinx-autobuild --host 0.0.0.0 docs docs_build
```

## Release

1. Create and push git tag
```shell
git tag v0.3.1
git push origin --tags
```

Confirm the version number
```shell
python -m setuptools_scm
```

## Publish in PyPI

```shell
pip install twine
python setup.py sdist bdist_wheel

# Test PyPI
python -m twine upload --repository testpypi dist/*
```
