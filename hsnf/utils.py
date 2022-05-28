from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias  # for Python<3.10

NDArrayInt: TypeAlias = npt.NDArray[np.int_]


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


def extgcd(a, b):
    """
    Extended Euclidean algorithm for ax + by = gcd(a, b)
    Return (gcd(a, b), x, y)
    """
    if b == 0:
        return (a, 1, 0)
    else:
        g, xx, yy = extgcd(b, a % b)
        x = yy
        y = xx - yy * (a // b)
        return (g, x, y)


def eratosthenes(n: int) -> dict[int, int]:
    sieve = [1 for _ in range(n + 1)]
    for d in range(2, n + 1):
        if sieve[d] != 1:
            continue
        for i in range(1, n // d + 1):
            sieve[d * i] = d

    factors = {}  # type: ignore
    while n > 1:
        p = sieve[n]
        n //= p
        factors[p] = factors.get(p, 0) + 1

    return factors


def crt(b1: NDArrayInt, b2: NDArrayInt, m1: int, m2: int):
    """
    Solve Chinese remainder theorem
        x mod m1 = b1
        x mod m2 = b2
    Return (r, lcm(m1, m2)) s.t. x mod lcm(m1, m2) == r
    If no solution exists, return None.
    """
    g, x, y = extgcd(m1, m2)  # m1 * x + m2 * y == g
    if not np.allclose(np.mod(b1 - b2, g), 0):
        return None
    lcm = m1 * m2 // g
    tmp = np.mod(((b2 - b1) / g * x).astype(int), m2 // g)
    r = np.mod(b1 + m1 * tmp, lcm)
    return (r, lcm)


def crt_on_list(offsets_and_modulo: list[tuple[NDArrayInt, int]]):
    r, lcm = offsets_and_modulo[0]
    for i in range(1, len(offsets_and_modulo)):
        b1, m1 = offsets_and_modulo[i - 1]
        b2, m2 = offsets_and_modulo[i]
        r, lcm = crt(b1, b2, m1, m2)

    return r, lcm


def get_triangular_rank(A):
    """
    Return rank of triangular integer matrix
    """
    return np.count_nonzero(np.diagonal(A))
