from typing import Optional

import numpy as np
from scipy.linalg import solve_triangular

from hsnf import (
    column_style_hermite_normal_form,
    row_style_hermite_normal_form,
    smith_normal_form,
)
from hsnf.lattice import compute_dual
from hsnf.utils import (
    NDArrayInt,
    crt_on_list,
    eratosthenes,
    extgcd,
    get_triangular_rank,
)


def solve_integer_linear_system(A: NDArrayInt, b: NDArrayInt):
    r"""
    For given :math:`\mathbf{A} \in \mathbb{Z}^{m \times n}` and :math:`\mathbf{b} \in \mathbb{Z}^{m}`, solve integer linear system :math:`\mathbf{Ax} = \mathbf{b}` in :math:`\mathbf{x} \in \mathbb{Z}^{n}`.
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
    For given :math:`\mathbf{A} \in \mathbb{Z}^{m \times n}` and :math:`\mathbf{b} \in \mathbb{Z}^{m}`, solve Frobenius congruent :math:`\mathbf{Ax} \equiv \mathbf{b} \, (\mathrm{mod}\, \mathbb{R}/\mathbb{Z})` for :math:`\mathbf{x} \in \mathbb{R}^{n}`.
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
        ``basis_Z[i, :]`` is a solution of :math:`\mathbf{Ax} \equiv \mathbf{0} \, (\mathrm{mod} \, \mathbb{Z})`
    basis_R: array, (n - rank, n)
        ``basis_R[i, :]`` is a solution of :math:`\mathbf{Ax} \equiv \mathbf{0} \, (\mathrm{mod}\, \mathbb{R}/\mathbb{Z})`
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


def solve_modular_integer_linear_system(A: NDArrayInt, b: NDArrayInt, q: int):
    r"""
    For given :math:`\mathbf{A} \in \mathbb{Z}^{m \times n}` and :math:`\mathbf{b} \in \mathbb{Z}^{m}`, solve modular integer linear system :math:`\mathbf{Ax} \equiv \mathbf{b} \, (\mathrm{mod} \, q)` in :math:`\mathbf{x} \in \mathbb{Z}^{n}`.
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
    q: int
        Modulo

    Returns
    -------
    basis: array, (r, n)
        ``basis[i, :]`` is a solution of :math:`\mathbf{Ax} \equiv \mathbf{0} \, (\mathrm{mod} \, q)`
    x_special: array, (n, )
        Special solution :math:`\mathbf{x}_{\mathrm{special}}`
    """
    D, L, R = smith_normal_form(A)
    rank = get_triangular_rank(D)
    Lb = np.dot(L, b)

    # Special solution can be constructed by solving Ax=b in modulo prime powers and use Chinese remainder theorem.
    y_special = _solve_modular_integer_linear_system_special(D, Lb, q, rank)
    x_special = np.mod(np.dot(R, y_special), q)
    assert np.allclose(np.mod(np.dot(A, x_special) - b, q), 0)

    # General solution of Ax=0 (mod q)
    basis = _solve_modular_integer_linear_system_general(A, q)
    assert np.allclose(np.mod(np.dot(basis, A.T), q), 0)

    return basis, x_special


def _solve_modular_integer_linear_system_general(A: NDArrayInt, q: int):
    """
    Calculate general solutions of Ax=0 (mod q)

    Implementation Note
    -------------------
    A: (m, n)

    Consider q-ary lattices
        Lambda_{q}^T := { x | Ax=0 (mod q) }
        Lambda_{q}   := { x | x = A^T s (mod q) for some s }
    They are dual each other:
        Lambda_{q}^T = q * dual(Lambda_{q}).
    A base of Lambda_{q} can be taken as A[i, :] (i=1...m) and q I[j, :] (j=1...n)

    Ref: https://cseweb.ucsd.edu/classes/wi12/cse206A-a/lec4.pdf
    """
    redundant_compliment = np.concatenate([A, q * np.eye(A.shape[1]).astype(int)], axis=0)

    # Choose independent vectors by HNF
    compliment, _ = row_style_hermite_normal_form(redundant_compliment)
    rank = get_triangular_rank(compliment)
    compliment = compliment[:rank, :]

    basis = q * compute_dual(compliment, row_wise=True)
    basis = np.mod(np.around(basis).astype(int), q)

    # Remove zero vectors
    used = np.count_nonzero(basis, axis=1) > 0
    basis = basis[used]

    return basis


def _solve_modular_integer_linear_system_special(D: NDArrayInt, Lb: NDArrayInt, q: int, rank: int):
    """
    Calculate a special solution of Dy = Lb (mod q) where D is SNF.
    """
    factors = eratosthenes(q)
    specials = []
    for p, l in factors.items():
        sol = _solve_modular_integer_linear_system_special_prime_power(D, Lb, p**l, rank)
        if sol is None:
            return None
        assert np.allclose(np.mod(D @ sol - Lb, p**l), 0)
        specials.append((sol, p**l))

    y_special, _ = crt_on_list(specials)

    return y_special


def _solve_modular_integer_linear_system_special_prime_power(
    D: NDArrayInt, Lb: NDArrayInt, prime_power: int, rank: int
):
    """
    Calculate a special solution of Dy = Lb (mod p^l) where D is SNF and p is prime.
    """
    y = np.zeros(D.shape[1]).astype(int)
    # Solve D[i, i] * y[i] = Lb[i] (mod prime_power)
    for i in range(rank):
        g, yi, _ = extgcd(D[i, i], prime_power)
        if Lb[i] % g != 0:
            return None
        y[i] = yi * Lb[i] // g

    y = np.mod(y, prime_power)
    return y


def remainder1_with_denominator(arr: NDArrayInt, denominator: int) -> NDArrayInt:
    """
    return arr (mod 1)
    """
    arr_int = np.around(arr * denominator).astype(int)
    arr_int_mod = np.remainder(arr_int, denominator)
    arr_mod = arr_int_mod / denominator
    return arr_mod
