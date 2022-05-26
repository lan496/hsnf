from setuptools_scm import get_version

from hsnf.Z_module import (  # noqa: F401
    column_style_hermite_normal_form,
    row_style_hermite_normal_form,
    smith_normal_form,
)

__version__ = get_version(root="..", relative_to=__file__)
