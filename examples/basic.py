import numpy as np

from hsnf import (
    column_style_hermite_normal_form,
    row_style_hermite_normal_form,
    smith_normal_form,
)

if __name__ == "__main__":
    # fmt: off
    M = np.array(
        [
            [-6, 111, -36, 6],
            [5, -672, 210, 74],
            [0, -255, 81, 24],
        ]
    )
    # fmt: on

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
