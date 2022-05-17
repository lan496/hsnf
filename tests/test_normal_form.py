# Copyright (c) 2019 Kohei Shinohara
# Distributed under the terms of the MIT License.
import numpy as np
import pytest

from hsnf import (
    column_style_hermite_normal_form,
    row_style_hermite_normal_form,
    smith_normal_form,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def is_unimodular(A) -> bool:
    """
    Return True if A is unimodular matrix
    """
    if not np.isclose(np.abs(np.linalg.det(A)), 1):
        return False

    A_inv = np.around(np.linalg.inv(A))
    return np.allclose(np.eye(A.shape[0]), np.dot(A, A_inv))


def verify_snf(M, D, L, R):
    """
    Assert (D, L, R) is SNF decomposition of M
    """
    D_re = np.dot(L, np.dot(M, R))
    assert np.allclose(D_re, D)

    D_diag = np.diagonal(D)
    rank = np.count_nonzero(D_diag)
    assert rank == np.count_nonzero(D)

    for i in range(rank - 1):
        assert D_diag[i + 1] % D_diag[i] == 0

    assert is_unimodular(L)
    assert is_unimodular(R)


def test_snf_small():
    M = np.array([[2, 4, 4], [-6, 6, 12], [10, 4, 16]])
    D, L, R = smith_normal_form(M)
    assert np.allclose(D, np.diag([2, 2, 156]))
    verify_snf(M, D, L, R)


def test_snf_random(rng):
    # test for square and non-square matrices
    list_size = [(100, 3, 7), (100, 11, 5), (100, 13, 13)]

    for size in list_size:
        X = rng.integers(-1, 1, size=size)
        for i in range(size[0]):
            D, L, R = smith_normal_form(X[i])
            verify_snf(X[i], D, L, R)


def verify_row_style_hnf(M, H, L):
    H_re = np.dot(L, M)

    assert np.allclose(H_re, H)
    assert np.allclose(H, np.triu(H))

    for s in range(min(H.shape)):
        assert H[s, s] >= 0
        if (s + 1 < H.shape[0]) and (H[s, s] > 0):
            assert np.max(H[(s + 1) :, s]) < H[s, s]

    assert is_unimodular(L)


def verify_column_style_hnf(M, H, R):
    H_re = np.dot(M, R)

    assert np.allclose(H_re, H)
    assert np.allclose(H, np.tril(H))
    for s in range(min(H.shape)):
        assert H[s, s] >= 0
        if (s > 0) and (H[s, s] > 0):
            assert np.max(H[s, :s]) < H[s, s]

    assert is_unimodular(R)


def test_hnf_small():
    A = np.array(
        [
            [2, 3, 6, 2],
            [5, 6, 1, 6],
            [8, 3, 1, 1],
        ]
    )
    H, L = row_style_hermite_normal_form(A)
    H_expect = np.array(
        [
            [1, 0, 50, -11],
            [0, 3, 28, -2],
            [0, 0, 61, -13],
        ]
    )
    assert np.allclose(H, H_expect)
    verify_row_style_hnf(A, H, L)


def test_hnf_random(rng):
    # test for square and non-square matrices
    list_size = [(100, 3, 7), (100, 11, 5), (100, 13, 13)]

    for size in list_size:
        X = rng.integers(-1, 1, size=size)
        for i in range(size[0]):
            H, L = row_style_hermite_normal_form(X[i])
            verify_row_style_hnf(X[i], H, L)

            H, R = column_style_hermite_normal_form(X[i])
            verify_column_style_hnf(X[i], H, R)


@pytest.mark.skip
def test_hnf_uniqueness():
    A1 = np.array([[3, 3, 1, 4], [0, 1, 0, 0], [0, 0, 19, 16], [0, 0, 0, 3]])
    H1_row_exp = np.array([[3, 0, 1, 1], [0, 1, 0, 0], [0, 0, 19, 1], [0, 0, 0, 3]])
    H1_row_act, _ = row_style_hermite_normal_form(A1)
    assert np.allclose(H1_row_act, H1_row_exp)

    A2 = np.array(
        [
            [0, 0, 5, 0, 1, 4],
            [0, 0, 0, -1, -4, 99],
            [0, 0, 0, 20, 19, 16],
            [0, 0, 0, 0, 2, 1],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    H2_row_exp = np.array(
        [
            [0, 0, 5, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    H2_row_act, _ = row_style_hermite_normal_form(A2)
    assert np.allclose(H2_row_act, H2_row_exp)

    A3 = np.array([[2, 3, 6, 2], [5, 6, 1, 6], [8, 3, 1, 1]])
    H3_row_exp = np.array([[1, 0, 50, -11], [0, 3, 28, -2], [0, 0, 61, -13]])
    H3_row_act, _ = row_style_hermite_normal_form(A3)
    assert np.allclose(H3_row_act, H3_row_exp)
