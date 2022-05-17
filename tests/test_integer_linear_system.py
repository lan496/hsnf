import numpy as np

from hsnf.integer_system import solve_frobenius_congruent, solve_integer_linear_system


def test_frobenius_congruent():
    A = np.array([[6, 4, 10], [-1, 1, -5]])
    b = np.array([4, 11])
    basis_Z, basis_R, x_special = solve_frobenius_congruent(A, b)

    assert np.allclose(np.remainder(A @ x_special, 1), np.remainder(b, 1))
    assert np.allclose(np.remainder(basis_Z @ A.T, 1), 0)
    assert np.allclose(basis_R @ A.T, 0)


def test_interger_linear_system():
    A = np.array([[6, 4, 10], [-1, 1, -5]])
    b = np.array([4, 11])
    basis, x_special = solve_integer_linear_system(A, b)

    assert np.allclose(A @ x_special, b)
    assert np.allclose(np.remainder(basis @ A.T, 1), 0)
