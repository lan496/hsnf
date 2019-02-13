# Copyright (c) 2019 Kohei Shinohara
# Distributed under the terms of the MIT License.

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
        # test for square and non-square matrices
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
        # test for square and non-square matrices
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


    @unittest.skip
    def test_hnf_uniqueness(self):
        A1 = np.array([
            [3, 3, 1, 4],
            [0, 1, 0, 0],
            [0, 0, 19, 16],
            [0, 0, 0, 3]
        ])
        H1_row_exp = np.array([
            [3, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 19, 1],
            [0, 0, 0, 3]
        ])
        H1_row_act, _ = row_style_hermite_normal_form(A1)
        self.assertTrue(np.allclose(H1_row_act, H1_row_exp))

        A2 = np.array([
            [0, 0, 5, 0, 1, 4],
            [0, 0, 0, -1, -4, 99],
            [0, 0, 0, 20, 19, 16],
            [0, 0, 0, 0, 2, 1],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0]
        ])
        H2_row_exp = np.array([
            [0, 0, 5, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        H2_row_act, _ = row_style_hermite_normal_form(A2)
        print(H2_row_act)
        self.assertTrue(np.allclose(H2_row_act, H2_row_exp))

        A3 = np.array([
            [2, 3, 6, 2],
            [5, 6, 1, 6],
            [8, 3, 1, 1]
        ])
        H3_row_exp = np.array([
            [1, 0, 50, -11],
            [0, 3, 28, -2],
            [0, 0, 61, -13]
        ])
        H3_row_act, _ = row_style_hermite_normal_form(A3)
        self.assertTrue(np.allclose(H3_row_act, H3_row_exp))

    def verify_snf(self, M, D, L, R):
        D_re = np.dot(L, np.dot(M, R))
        self.assertTrue(np.array_equal(D_re, D))

        D_diag = np.diagonal(D)
        rank = np.count_nonzero(D_diag)
        self.assertEqual(np.count_nonzero(D) - rank, 0)

        for i in range(rank - 1):
            self.assertTrue(D_diag[i + 1] % D_diag[i] == 0)

        self.is_unimodular(L)
        self.is_unimodular(R)

    def verify_row_style_hnf(self, M, H, L):
        H_re = np.dot(L, M)

        self.assertTrue(np.array_equal(H_re, H))
        self.assertTrue(np.allclose(H, np.triu(H)))
        for s in range(min(H.shape)):
            self.assertTrue(H[s, s] >= 0)
            if (s + 1 < H.shape[0]) and (H[s, s] > 0):
                self.assertTrue(np.max(H[(s + 1):, s]) < H[s, s])

        self.is_unimodular(L)

    def verify_column_style_hnf(self, M, H, R):
        H_re = np.dot(M, R)

        self.assertTrue(np.array_equal(H_re, H))
        self.assertTrue(np.allclose(H, np.tril(H)))
        for s in range(min(H.shape)):
            self.assertTrue(H[s, s] >= 0)
            if (s > 0) and (H[s, s] > 0):
                self.assertTrue(np.max(H[s, :s]) < H[s, s])

        self.is_unimodular(R)

    def is_unimodular(self, A):
        self.assertAlmostEqual(np.abs(np.linalg.det(A)), 1)

        A_inv = np.around(np.linalg.inv(A))
        self.assertTrue(np.allclose(np.eye(A.shape[0]), np.dot(A, A_inv)))


if __name__ == '__main__':
    unittest.main()
