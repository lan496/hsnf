# hsnf
[![testing](https://github.com/lan496/hsnf/actions/workflows/testing.yml/badge.svg?branch=master)](https://github.com/lan496/hsnf/actions/workflows/testing.yml)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hsnf)
[![PyPI version](https://badge.fury.io/py/hsnf.svg)](https://badge.fury.io/py/hsnf)
![PyPI - Downloads](https://img.shields.io/pypi/dm/hsnf)
<!--![GitHub all releases](https://img.shields.io/github/downloads/lan496/hsnf/total) -->

Computing Hermite normal form and Smith normal form with transformation matrices.

- Github: https://github.com/lan496/hsnf
- PyPI: https://pypi.org/project/hsnf/

## Usage

```python
import numpy as np
from hsnf import column_style_hermite_normal_form, row_style_hermite_normal_form, smith_normal_form

# Integer matrix to be decomposed
M = np.array(
    [
        [-6, 111, -36, 6],
        [5, -672, 210, 74],
        [0, -255, 81, 24],
    ]
)

# Smith normal form
D, L, R = smith_normal_form(M)
"""
D = array([
[   1    0    0    0]
[   0    3    0    0]
[   0    0 2079    0]])
"""
assert np.allclose(L @ M @ R, D)
assert np.around(np.abs(np.linalg.det(L))) == 1  # unimodular
assert np.around(np.abs(np.linalg.det(R))) == 1  # unimodular

# Row-style hermite normal form
H, L = row_style_hermite_normal_form(M)
"""
H = array([
[     1      0    420  -2522]
[     0      3   1809 -10860]
[     0      0   2079 -12474]])
"""
assert np.allclose(L @ M, H)
assert np.around(np.abs(np.linalg.det(L))) == 1  # unimodular

# Column-style hermite normal form
H, R = column_style_hermite_normal_form(M)
"""
H = array([
[   3    0    0    0]
[   0    1    0    0]
[1185  474 2079    0]])
"""
assert np.allclose(np.dot(M, R), H)
assert np.around(np.abs(np.linalg.det(R))) == 1  # unimodular
```

## Installation

hsnf works with Python3.8+ and can be installed via PyPI:

```shell
pip install hsnf
```

or in local:
```shell
git clone git@github.com:lan496/hsnf.git
cd hsnf
pip install .
```

## References
- http://www.dlfer.xyz/post/2016-10-27-smith-normal-form/
  - I appreciate Dr. D. L. Ferrario's instructive blog post and his approval for referring his scripts.
- [CSE206A: Lattices Algorithms and Applications (Spring 2014)](https://cseweb.ucsd.edu/classes/sp14/cse206A-a/index.html)
