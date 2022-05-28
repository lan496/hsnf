import numpy as np

from hsnf.integer_system import (
    solve_frobenius_congruent,
    solve_integer_linear_system,
    solve_modular_integer_linear_system,
)


def test_frobenius_congruent():
    A = np.array([[6, 4, 10], [-1, 1, -5]])
    b = np.array([4, 11])
    basis_Z, basis_R, x_special = solve_frobenius_congruent(A, b)

    assert np.allclose(np.remainder(A @ x_special, 1), np.remainder(b, 1))
    assert np.allclose(np.remainder(basis_Z @ A.T, 1), 0)
    assert np.allclose(basis_R @ A.T, 0)


def test_integer_linear_system():
    A = np.array([[6, 4, 10], [-1, 1, -5]])
    b = np.array([4, 11])
    basis, x_special = solve_integer_linear_system(A, b)

    assert np.allclose(A @ x_special, b)
    assert np.allclose(np.remainder(basis @ A.T, 1), 0)


def test_modular_integer_linear_system():
    # Example adapted from https://math.stackexchange.com/questions/2556129/how-to-solve-a-system-of-linear-equations-modulo-n
    A = np.array([[4, -10], [7, 2]])
    b = np.array([8, 5])
    q = 20
    basis, x_special = solve_modular_integer_linear_system(A, b, q)
    assert np.allclose(x_special, np.array([7, 8]))
    assert np.allclose(basis, np.array([[0, 10]]))
