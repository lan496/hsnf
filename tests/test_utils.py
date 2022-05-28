import numpy as np

from hsnf.utils import crt_on_list, eratosthenes, get_triangular_rank


def test_rank():
    A = np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
        ]
    )
    assert get_triangular_rank(A) == 2


def test_eratosthenes():
    actual = eratosthenes(24)
    expect = {2: 3, 3: 1}
    assert actual == expect


def test_crt():
    # x = 0 (mod 3), x = 3 (mod 4), x = 4 (mod 5)
    actual = crt_on_list(
        [
            (np.array([0]), 3),
            (np.array([3]), 4),
            (np.array([4]), 5),
        ]
    )

    # Solution: x = 39 (mod 40)
    assert actual[0] == 19
    assert actual[1] == 20  # lcm
