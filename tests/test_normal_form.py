import unittest

import numpy as np

from hsnf import (
    smith_normal_form,
    row_style_hermite_normal_form,
    column_style_hermite_normal_form,
)


class TestNormalForm(unittest.TestCase):

    def setUp(self):
        self.random_state = 0

    def test_snf(self):
        np.random.seed(self.random_state)
        list_size = [
            (1000, 3, 7),
            (1000, 11, 5),
            (1000, 13, 13)
        ]

        for size in list_size:
            X = np.random.randint(-1, 1, size=size)
            for i in range(size[0]):
                D, L, R = smith_normal_form(X[i])
                self.verify_snf(X[i], D, L, R)

    def test_hnf(self):
        np.random.seed(self.random_state)
        list_size = [
            (1000, 3, 7),
            (1000, 11, 5),
            (1000, 13, 13)
        ]

        for size in list_size:
            X = np.random.randint(-1, 1, size=size)
            for i in range(size[0]):
                H, L = row_style_hermite_normal_form(X[i])
                self.verify_row_style_hnf(X[i], H, L)

                H, R = column_style_hermite_normal_form(X[i])
                self.verify_column_style_hnf(X[i], H, R)

    def verify_snf(self, M, D, L, R):
        D_re = np.dot(L, np.dot(M, R))
        self.assertEqual(np.around(np.abs(np.linalg.det(L))), 1)
        self.assertEqual(np.around(np.abs(np.linalg.det(R))), 1)
        self.assertTrue(np.array_equal(D_re, D))

        D_diag = np.diagonal(D)
        rank = np.count_nonzero(D_diag)
        self.assertEqual(np.count_nonzero(D) - rank, 0)

        for i in range(rank - 1):
            self.assertTrue(D_diag[i + 1] % D_diag[i] == 0)

    def verify_row_style_hnf(self, M, H, L):
        H_re = np.dot(L, M)
        self.assertEqual(np.around(np.abs(np.linalg.det(L))), 1)
        self.assertTrue(np.array_equal(H_re, H))
        self.assertTrue(np.allclose(H, np.triu(H)))

    def verify_column_style_hnf(self, M, H, R):
        H_re = np.dot(M, R)
        self.assertEqual(np.around(np.abs(np.linalg.det(R))), 1)
        self.assertTrue(np.array_equal(H_re, H))
        self.assertTrue(np.allclose(H, np.tril(H)))


if __name__ == '__main__':
    unittest.main()
