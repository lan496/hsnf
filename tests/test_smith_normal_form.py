import unittest

import numpy as np

from hsnf.smith_normal_form import smith_normal_form


class TestSmithNormalForm(unittest.TestCase):

    def test_snf(self):
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
                [-6, 111, -36, 6],
                [5, -672, 210, 74],
                [0, -255, 81, 24],
                [-7, 255, -81, -10]
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
            np.diag([1, 8]),
            np.diag([2, 6, 12]),
            np.array([
                [4, 0, 0],
                [0, 12, 0]
            ]),
            np.diag([1, 3, 21, 0]),
            np.diag([1, 4, 4]),
            np.diag([1, 2, 2])
        ]

        for M, expected in zip(list_matrix, list_expected):
            D, L, R = smith_normal_form(M)
            D_re = np.dot(L, np.dot(M, R))
            self.assertAlmostEqual(np.linalg.det(L) ** 2, 1)
            self.assertAlmostEqual(np.linalg.det(R) ** 2, 1)
            self.assertTrue(np.array_equal(D_re, D))

    def test_snf_for_random_matrix(self):
        random_state = 0
        size = (10000, 3, 7)
        np.random.seed(random_state)

        X = np.random.random_integers(-1, 1, size=size)
        for i in range(size[0]):
            D, L, R = smith_normal_form(X[i])
            self.verify_snf(X[i], D, L, R)

    def verify_snf(self, M, D, L, R):
        D_re = np.dot(L, np.dot(M, R))
        self.assertAlmostEqual(np.linalg.det(L) ** 2, 1)
        self.assertAlmostEqual(np.linalg.det(R) ** 2, 1)
        self.assertTrue(np.array_equal(D_re, D))
        print(M)
        print(D)
        self.assertEqual(np.count_nonzero(D) - np.count_nonzero(np.diagonal(D)), 0)


if __name__ == '__main__':
    unittest.main()
