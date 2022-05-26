from math import gcd

import numpy as np

from hsnf import row_style_hermite_normal_form
from hsnf.utils import NDArrayInt, get_triangular_rank


def to_row_wise(lattice, row_wise: bool):
    if row_wise:
        return lattice
    else:
        return lattice.T


def equivalent(lattice1: NDArrayInt, lattice2: NDArrayInt, row_wise: bool = True) -> bool:
    """
    Return if and only if given two lattices are equivalent.
    Assume both lattices are full rank.

    Two lattices are equivalent if and only if they have the same Hermite normal forms.

    Parameters
    ----------
    lattice1: array, (n, n)
        If ``row_wise=True``, ``lattice1[i, :]`` is the i-th basis vector of the lattice.
        Otherwise ``lattice1[:, i]`` is.
    lattice2: array, (n, n)
        If ``row_wise=True``, ``lattice2[i, :]`` is the i-th basis vector of the lattice
        Otherwise ``lattice2[:, i]`` is.
    row_wise:
        If true, basis vectors are aligned in row wise, otherwise in column wise.

    """
    l1 = to_row_wise(lattice1, row_wise)
    l2 = to_row_wise(lattice2, row_wise)

    H1, _ = row_style_hermite_normal_form(l1)
    H2, _ = row_style_hermite_normal_form(l2)

    # If two HNFs are equal, the two lattices are equivalent
    return np.allclose(H1, H2)


def compute_union(lattice1: NDArrayInt, lattice2: NDArrayInt, row_wise: bool = True):
    r"""
    Return the smallest lattice containing both lattice1 and lattice2

    The union of two lattices with basis vectors :math:`\mathbf{A}` and :math:`\mathbf{B}` is computed by a Hermite normal form of the concatenated matrices :math:`[\mathbf{A} | \mathbf{B}]`.

    Parameters
    ----------
    lattice1: array, (n, n)
        If ``row_wise=True``, ``lattice1[i, :]`` is the i-th basis vector of the lattice.
        Otherwise ``lattice1[:, i]`` is.
    lattice2: array, (n, n)
        If ``row_wise=True``, ``lattice2[i, :]`` is the i-th basis vector of the lattice
        Otherwise ``lattice2[:, i]`` is.
    row_wise:
        If true, basis vectors are aligned in row wise, otherwise in column wise.
    """
    l1 = to_row_wise(lattice1, row_wise)
    l2 = to_row_wise(lattice2, row_wise)

    H, _ = row_style_hermite_normal_form(np.concatenate([l1, l2], axis=0))
    rank = get_triangular_rank(H)

    union = H[:rank, :]

    if not row_wise:
        union = union.T

    return union


def compute_dual(lattice, row_wise: bool = True):
    """
    Return basis of a dual lattice.

    Parameters
    ----------
    lattice: array, (n, n)
        If ``row_wise=True``, ``lattice[i, :]`` is the i-th basis vector of the lattice.
        Otherwise ``lattice[:, i]`` is.
    row_wise:
        If true, basis vectors are aligned in row wise, otherwise in column wise.
    """
    lat = to_row_wise(lattice, row_wise)
    d = np.linalg.inv(lat @ lat.T) @ lat
    if not row_wise:
        d = d.T
    return d


def compute_intersection(lattice1: NDArrayInt, lattice2: NDArrayInt, row_wise: bool = True):
    """
    Return intersection lattice of lattice1 and lattice2.

    Let a dual of lattice :math:`L` be :math:`\tilde{L}`.
    The dual of intersection of lattices :math:`L` and :math:`L'` is a union of dual lattices of :math:`L` and :math:`L'`:

    .. math::
        \\widetilde{ L \\cap L' } = \\tilde{L} \\cup \\tilde{L'}

    Parameters
    ----------
    lattice1: array, (n, n)
        If ``row_wise=True``, ``lattice1[i, :]`` is the i-th basis vector of the lattice.
        Otherwise ``lattice1[:, i]`` is.
    lattice2: array, (n, n)
        If ``row_wise=True``, ``lattice2[i, :]`` is the i-th basis vector of the lattice
        Otherwise ``lattice2[:, i]`` is.
    row_wise:
        If true, basis vectors are aligned in row wise, otherwise in column wise.
    """
    l1 = to_row_wise(lattice1, row_wise)
    l2 = to_row_wise(lattice2, row_wise)

    denom1 = int(np.around(np.linalg.det(l1))) ** 2
    denom2 = int(np.around(np.linalg.det(l2))) ** 2
    denom = denom1 * denom2 // gcd(denom1, denom2)

    # dual(intersection(l1, l2)) = union(dual(l1), dual(l2))
    d1 = np.around(compute_dual(l1) * denom).astype(int)
    d2 = np.around(compute_dual(l2) * denom).astype(int)
    dunion = compute_union(d1, d2)
    ret = np.around(compute_dual(dunion) * denom).astype(int)

    if not row_wise:
        ret = ret.T

    return ret
