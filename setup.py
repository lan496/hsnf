#!/usr/bin/env python
# https://github.com/kennethreitz/setup.py/blob/master/setup.py

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = "hsnf"
DESCRIPTION = "Computing Hermite normal form and Smith normal form."
URL = "https://github.com/lan496/hsnf"
AUTHOR = "Kohei Shinohara"
EMAIL = ""
REQUIRES_PYTHON = ">=3.8.0"


# What packages are required for this module to be executed?
REQUIRED = ["setuptools", "setuptools_scm", "wheel", "numpy>=1.20.1", "scipy", "typing_extensions"]

# What packages are optional?
EXTRAS = {
    "dev": [
        "pytest",
        "pytest-cov",
        "pre-commit",
        "black",
        "mypy",
        "flake8",
        "pyupgrade",
    ],
    "docs": [
        "sphinx",
        "sphinx-autobuild",
        "sphinxcontrib-bibtex",
        "myst-parser",
        "sphinx-book-theme",
    ],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Where the magic happens:
setup(
    name=NAME,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={
        "hsnf": ["py.typed"],
    },
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    # numpy: https://github.com/numpy/numpy/issues/2434
    setup_requires=["setuptools_scm", "numpy"],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    test_suite="tests",
    zip_safe=False,
    use_scm_version=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
