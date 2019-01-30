import unittest

import numpy as np

from hsnf.hermite_normal_form import hermite_normal_form


class TestHermiteNormalForm(unittest.TestCase):

    def test_hnf(self):
        list_matrix = [
            np.array([
                [2, 0],
                [1, 4]
            ]),
            np.array([
                [2, 4, 4],
                [-6, 6, 12],
                [10, -4, -16]
            ]),
            np.array([
                [8, 4, 8],
                [4, 8, 4]
            ]),
            np.array([
                [3, -1, -1],
                [-1, 3, -1],
                [-1, -1, 3]
            ]),
            np.array([
                [1, 0, 0],
                [1, 2, 0],
                [0, 0, 2]
            ]),
        ]
        list_expected = [
            np.array([
                [1, 4],
                [0, 8]
            ]),
            np.array([
                [2, 4, 4],
                [0, 6, 12],
                [0, 0, 12]
            ]),
            np.array([
                [4, 8, 4],
                [0, 12, 0]
            ]),
            np.array([
                [1, -3, 1],
                [0, 4, -4],
                [0, 0, 4]
            ]),
            np.diag([1, 2, 2])
        ]

        for M, expected in zip(list_matrix, list_expected):
            H, L = hermite_normal_form(M)
            H_re = np.dot(L, M)
            self.assertAlmostEqual(np.linalg.det(L) ** 2, 1)
            self.assertTrue(np.array_equal(H_re, H))


if __name__ == '__main__':
    unittest.main()
