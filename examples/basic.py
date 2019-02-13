import numpy as np
from hsnf import smith_normal_form
from hsnf import row_style_hermite_normal_form, column_style_hermite_normal_form


if __name__ == '__main__':
    M = np.array([
        [-6, 111, -36, 6],
        [5, -672, 210, 74],
        [0, -255, 81, 24],
    ])
    D, L, R = smith_normal_form(M)
    assert np.allclose(np.dot(L, np.dot(M, R)), D)
    print(D)
    print()

    H, L = row_style_hermite_normal_form(M)
    assert np.allclose(np.dot(L, M), H)
    print(H)
    print()

    H, R = column_style_hermite_normal_form(M)
    assert np.allclose(np.dot(M, R), H)
    print(H)
