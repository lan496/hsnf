# hsnf
Compute Hermite normal form and Smith normal form

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
    [   1,    0,    0,    0],
    [   0,    3,    0,    0],
    [   0,    0, 2079,    0]])
L = array([
    [     0,      1,      0],
    [     0,   -279,      1],
    [     1, -28689,    100]])
R = array([
    [-149,   26, -186,    2],
    [  -1,    4,  -27,    2],
    [   0,   15, -101,    6],
    [   1,   -8,   54,    1]])
"""
np.dot(L, np.dot(M, R))  # equal to D, diagonal
np.abs(np.linalg.det(L))  # equal to 1
np.abs(np.linalg.det(R))  # equal to 1

# row-style hermite normal form
H, L = row_style_hermite_normal_form(M)
"""
H = array([
    [     1,    561,   -174,    -80],
    [     0,      3,   -270,   1614],
    [     0,      0,   2079, -12474]])
L = array([
    [  -1,   -1,    0],
    [  55,   66, -150],
    [-425, -510, 1159]])
"""
np.dot(L, M)  # equal to H, upper triangular
np.abs(np.linalg.det(L))  # equal to 1

# column-style hermite normal form
H, R = column_style_hermite_normal_form(M)
"""
H = array([
    [   3,    0,    0,    0],
    [ 577,    1,    0,    0],
    [ 255,  474, 2079,    0]])
R = array([
    [ -19, -155, -680,    2],
    [  -1,   -6,  -26,    2],
    [   0,    0,    1,    6],
    [   0,  -44, -193,    1]])
"""
np.dot(M, R)  # equal to H
np.abs(np.linalg.det(R))  # equal to 1
```

## References
- http://www.dlfer.xyz/post/2016-10-27-smith-normal-form/
