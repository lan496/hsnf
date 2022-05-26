from typing import Optional

import numpy as np
from scipy.linalg import solve_triangular

from hsnf import column_style_hermite_normal_form, smith_normal_form
from hsnf.utils import NDArrayInt, get_triangular_rank


def solve_integer_linear_system(A: NDArrayInt, b: NDArrayInt):
    r"""
    Solve integer linear system :math:`\mathbf{Ax} = \mathbf{b}` for :math:`\mathbf{x} \in \mathbb{Z}^{n}`.

    Let the rank of :math:`\mathbf{A}` as :math:`r`.
    General solutions are written as

    .. math::
        \{
            \mathbf{x}_{\mathrm{special}} + \sum_{i=0}^{r-1} a_{i} \cdot \mathrm{basis[i]}
            \mid
            a_{i} \in \mathbb{Z}
        \}.

    If no solution exists, return None.

    Parameters
    ----------
    A: array, (m, n)
        Integer coefficient matrix
    b: array, (m, )
        Integer offsets

    Returns
    -------
    basis: array, (rank, n)
        ``basis[i, :]`` is a solution of :math:`\mathbf{Ax}=\mathbf{0}`
    x_special: array, (n, )
        Special solution :math:`\mathbf{x}_{\mathrm{special}}`

    """
    H, R = column_style_hermite_normal_form(A)
    rank = get_triangular_rank(H)

    x_special = np.zeros(A.shape[1])
    x_special[:rank] = solve_triangular(H[:rank, :rank], b[:rank], lower=True)
    if not np.allclose(x_special, np.around(x_special)):
        return None
    x_special = np.dot(R, x_special)

    basis = R[:, :rank].T

    return basis, x_special


def solve_frobenius_congruent(
    A: NDArrayInt, b: Optional[NDArrayInt] = None, denominator: int = 1000000
):
    r"""
    Solve Frobenius congruent :math:`\mathbf{Ax} = \mathbf{b} \, (\mathrm{mod}\, \mathbb{Z})` for :math:`\mathbf{x} \in \mathbb{R}^{n}`.

    Let the rank of :math:`\mathbf{A}` as :math:`r`.
    General solutions are written as

    .. math::
        \{
            \mathbf{x}_{\mathrm{special}}
            + \sum_{i=0}^{r-1} a_{i} \cdot \mathrm{basis\_Z[i]}
            + \sum_{j=0}^{n-r-1} c_{j} \cdot \mathrm{basis\_R[j]}
            \mid
            a_{i} \in \mathbb{Z}, c_{j} \in \mathbb{R}
        \}.

    If no solution exists, return None.

    Parameters
    ----------
    A: array, (m, n)
        Integer coefficient matrix
    b: (Optional) array, (m, )
        Integer offsets
    denominator:
        (Optional) If specified, taking modulus as fraction with up to specified denominator

    Returns
    -------
    basis_Z: array, (rank, n)
        ``basis_Z[i, :]`` is a solution of :math:`\mathbf{Ax}=\mathbf{0}`
    basis_R: array, (n - rank, n)
        ``basis_R[i, :]`` is a solution of :math:`\mathbf{Ax}=\mathbf{0}`
    x_special: array, (n, )
        Special solution :math:`\mathbf{x}_{\mathrm{special}}`
    """
    D, P, Q = smith_normal_form(A)
    rank = get_triangular_rank(D)

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


def remainder1_with_denominator(arr: NDArrayInt, denominator: int) -> NDArrayInt:
    """
    return arr (mod 1)
    """
    arr_int = np.around(arr * denominator).astype(int)
    arr_int_mod = np.remainder(arr_int, denominator)
    arr_mod = arr_int_mod / denominator
    return arr_mod
