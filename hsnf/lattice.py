from math import gcd

import numpy as np
from numpy.typing import NDArray

from hsnf import row_style_hermite_normal_form


def to_row_wise(lattice, row_wise: bool):
    if row_wise:
        return lattice
    else:
        return lattice.T


def equivalent(lattice1: NDArray, lattice2: NDArray, row_wise: bool = True) -> bool:
    """
    Determine if two lattices are equivalent. If `row_wise` is True, lattice1[i, :] is the i-th basis vector.
    Assume both lattices are full rank.
    """
    l1 = to_row_wise(lattice1, row_wise)
    l2 = to_row_wise(lattice2, row_wise)

    H1, _ = row_style_hermite_normal_form(l1)
    H2, _ = row_style_hermite_normal_form(l2)

    # If two HNFs are equal, the two lattices are equivalent
    return np.allclose(H1, H2)


def compute_union(lattice1: NDArray, lattice2: NDArray, row_wise: bool = True):
    """
    Return the smallest lattice containing both lattice1 and lattice2
    """
    l1 = to_row_wise(lattice1, row_wise)
    l2 = to_row_wise(lattice2, row_wise)

    H, _ = row_style_hermite_normal_form(np.concatenate([l1, l2], axis=0))
    rank = np.count_nonzero(H.diagonal())

    union = H[:rank, :]

    if not row_wise:
        union = union.T

    return union


def compute_dual(lattice, row_wise: bool = True):
    """
    Return basis of dual lattice
    """
    lat = to_row_wise(lattice, row_wise)
    d = np.linalg.inv(lat @ lat.T) @ lat
    if not row_wise:
        d = d.T
    return d


def compute_intersection(lattice1: NDArray, lattice2: NDArray, row_wise: bool = True):
    """
    Return intersection of lattice1 and lattice2.
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