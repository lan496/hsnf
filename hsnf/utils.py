import numpy as np


def get_nonzero_min_abs(A, i1, i2, j1, j2):
    """
    return idx = argmin_{i, j} abs(A[i, j]) s.t. (i1 <= i < i2 and j1 <= j < j2 and A[i, j] != 0)
    if failed, return (None, None)
    """
    idx = (None, None)
    valmin = None

    for i in range(i1, i2):
        for j in range(j1, j2):
            if A[i, j] == 0:
                continue
            if (valmin is None) or (np.abs(A[i, j]) < valmin):
                idx = (i, j)
                valmin = np.abs(A[i, j])
    return idx


def get_nonzero_min_abs_full(A, s):
    """
    return idx = argmin_{i, j} abs(A[i, j]) s.t. (i >= s and j >= s and A[i, j] != 0)
    if failed, return (None, None)
    """
    return get_nonzero_min_abs(A, s, A.shape[0], s, A.shape[1])


def get_nonzero_min_abs_row(A, i1, j1):
    """
    return idx = argmin_{i, j} abs(A[i, j]) s.t. (i >= i1 and j == j1 and A[i, j] != 0)
    if failed, return (None, None)
    """
    return get_nonzero_min_abs(A, i1, A.shape[0], j1, j1 + 1)


def get_nonzero_min_abs_column(A, i1, j1):
    """
    return idx = argmin_{i, j} abs(A[i, j]) s.t. (i == i1 and j >= j1 and A[i, j] != 0)
    if failed, return (None, None)
    """
    return get_nonzero_min_abs(A, i1, i1 + 1, j1, A.shape[1])
