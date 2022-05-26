import numpy as np

from hsnf.utils import get_triangular_rank


def test_rank():
    A = np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
        ]
    )
    assert get_triangular_rank(A) == 2
