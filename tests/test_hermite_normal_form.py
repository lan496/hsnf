import unittest

import numpy as np

from hsnf.hermite_normal_form import (
    row_style_hermite_normal_form,
    column_style_hermite_normal_form,
)


class TestHermiteNormalForm(unittest.TestCase):

    def setUp(self):
        self.list_matrix = [
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
        self.list_row_style_expected = [
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
        self.list_column_style_expected = [H.T for H in self.list_row_style_expected]

    def test_row_style_hnf(self):
        for M, expected in zip(self.list_matrix, self.list_row_style_expected):
            H, L = row_style_hermite_normal_form(M)
            self.verify_row_style_hnf(M, H, L)

    def test_column_style_hnf(self):
        for M, expected in zip(self.list_matrix, self.list_column_style_expected):
            H, R = column_style_hermite_normal_form(M)
            self.verify_column_style_hnf(M, H, R)

    def test_snf_for_random_matrix(self):
        random_state = 0
        list_size = [
            (1000, 3, 7),
            (1000, 11, 5),
            (1000, 13, 13)
        ]
        np.random.seed(random_state)

        for size in list_size:
            X = np.random.randint(-1, 1, size=size)
            for i in range(size[0]):
                H, L = row_style_hermite_normal_form(X[i])
                self.verify_row_style_hnf(X[i], H, L)

                H, R = column_style_hermite_normal_form(X[i])
                self.verify_column_style_hnf(X[i], H, R)

    def verify_row_style_hnf(self, M, H, L):
        H_re = np.dot(L, M)
        self.assertAlmostEqual(np.linalg.det(L) ** 2, 1)
        self.assertTrue(np.array_equal(H_re, H))
        self.assertTrue(np.allclose(H, np.triu(H)))

    def verify_column_style_hnf(self, M, H, R):
        H_re = np.dot(M, R)
        self.assertAlmostEqual(np.linalg.det(R) ** 2, 1)
        self.assertTrue(np.array_equal(H_re, H))
        self.assertTrue(np.allclose(H, np.tril(H)))


if __name__ == '__main__':
    unittest.main()
