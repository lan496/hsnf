# hsnf
![Build Status](https://travis-ci.com/lan496/hsnf.svg?branch=master)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

(develop: ![Build Status](https://travis-ci.com/lan496/hsnf.svg?branch=develop))

Computing Hermite normal form and Smith normal form with transformation matrices.

## Usage
```example.py
import numpy as np
from hsnf import smith_normal_form
from hsnf import row_style_hermite_normal_form, column_style_hermite_normal_form

M = np.array([
    [-6, 111, -36, 6],
    [5, -672, 210, 74],
    [0, -255, 81, 24],
])

# smith normal form
D, L, R = smith_normal_form(M)
"""
D = array([
  [   1    0    0    0]
  [   0    3    0    0]
  [   0    0 2079    0]])
"""
np.dot(L, np.dot(M, R))  # equal to D, diagonal
np.abs(np.linalg.det(L))  # equal to 1
np.abs(np.linalg.det(R))  # equal to 1

# row-style hermite normal form
H, L = row_style_hermite_normal_form(M)
"""
H = array([
  [     1      0    420  -2522]
  [     0      3   1809 -10860]
  [     0      0   2079 -12474]])
"""
np.dot(L, M)  # equal to H, upper triangular
np.abs(np.linalg.det(L))  # equal to 1

# column-style hermite normal form
H, R = column_style_hermite_normal_form(M)
"""
H = array([
  [   3    0    0    0]
  [   0    1    0    0]
  [1185  474 2079    0]])
"""
np.dot(M, R)  # equal to H, lower-triangular
np.abs(np.linalg.det(R))  # equal to 1
```

## References
- http://www.dlfer.xyz/post/2016-10-27-smith-normal-form/

I appreciate D.L. Ferrario's instructive blog post and his approval for refering his scripts.
