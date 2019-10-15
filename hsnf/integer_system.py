from hsnf.Z_module import smith_normal_form

import numpy as np


def solve_frobenius_congruent(A, b=None, denominator=1000000):
    """
    solve Ax=b (mod Z^n)

    Parameters
    ----------
    A: array, (m, n)
    b: (Optional) array, (m, )
    denominator: (Optional), int

    Returns
    -------
    basis_Z: array, (rank, n)
    basis_R: array, (n - rank, n)
    x_special: array, (n, )

    general solution is written by
        x = x_special + (Z times basis_Z[0] + ...) + (R times basis_R[0] + ...)
    """
    D, P, Q = smith_normal_form(A)
    rank = np.count_nonzero(np.diagonal(D))

    D_pinv = np.zeros((Q.shape[1], rank))
    D_pinv[np.diag_indices(rank)] = 1 / D.diagonal()[:rank]
    # basis_Z[i, :] is the i-th general Z-solution of x
    basis_Z = np.dot(Q, D_pinv).T

    if rank < Q.shape[1]:
        y_R = np.zeros((Q.shape[1], Q.shape[1] - rank))
        for i in range(Q.shape[1] - rank):
            y_R[rank + i, i] = 1
        basis_R = np.dot(Q, y_R).T
    else:
        basis_R = None

    if b is None:
        return basis_Z, basis_R
    else:
        v = remainder1_with_denominator(np.dot(P, b), denominator)  # for avoiding numerical error
        if np.count_nonzero(v[rank:]) != 0:
            return basis_Z, basis_R, None

        y_special = np.zeros(A.shape[1])
        y_special[:rank] = v[:rank] / D.diagonal()[:rank]
        x_special = np.dot(Q, y_special)
        return basis_Z, basis_R, x_special


def remainder1_with_denominator(arr, denominator):
    """
    return arr (mod 1)
    """
    arr_int = np.around(arr * denominator).astype(int)
    arr_int_mod = np.remainder(arr_int, denominator)
    arr_mod = arr_int_mod / denominator
    return arr_mod
