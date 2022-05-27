from importlib.metadata import PackageNotFoundError, version

from hsnf.Z_module import (  # noqa: F401
    column_style_hermite_normal_form,
    row_style_hermite_normal_form,
    smith_normal_form,
)

# https://github.com/pypa/setuptools_scm/#retrieving-package-version-at-runtime
try:
    __version__ = version("hsnf")
except PackageNotFoundError:
    # package is not installed
    pass
