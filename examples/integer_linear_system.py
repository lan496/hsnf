import numpy as np
from scipy.linalg import solve_triangular
from hsnf import column_style_hermite_normal_form


def solve_integer_linear_system(A, b):
    H, R = column_style_hermite_normal_form(A)
    rank = np.count_nonzero(np.diagonal(H))
    print(H)

    x_special = np.zeros(A.shape[1])
    x_special[:rank] = solve_triangular(H[:rank, :rank], b[:rank], lower=True)
    if not np.allclose(x_special, np.around(x_special)):
        return None, None
    x_special = np.dot(R, x_special)

    basis = R[:, rank:]

    return x_special, basis


if __name__ == '__main__':
    A = np.array([
        [6, 4, 10],
        [-1, 1, -5]
    ])
    b = np.array([4, 11])
    x_special, basis = solve_integer_linear_system(A, b)
    print('special solution')
    print(x_special)
    print('general solution')
    print(basis)
