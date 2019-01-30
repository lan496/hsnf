import numpy as np


def change_sign_row(M, i):
    ret = M.copy()
    ret[i, :] *= -1
    return ret


def swap_rows(M, i, j):
    ret = M.copy()
    ret[i, :], ret[j, :] = M[j, :], M[i, :]
    return ret


def add_to_row(M, i, j, k):
    ret = M.copy()
    ret[i, :] += M[j, :] * k
    return ret


def get_min_abs(M, s):
    ret = s
    valmin = np.max(np.abs(M[s:, s])) + 1
    for i in range(s, M.shape[0]):
        if (M[i, s] != 0) and abs(M[i, s]) < valmin:
            ret = i
            valmin = abs(M[i, s])
    return ret


def _hnf(M, L, s):
    if (s == M.shape[0] - 1) or (s == M.shape[1] - 1):
        if M[s, s] < 0:
            M, L = change_sign_row(M, s), change_sign_row(L, s)
        return M, L

    row = get_min_abs(M, s)
    M, L = swap_rows(M, s, row), swap_rows(L, s, row)

    for i in range(s + 1, M.shape[0]):
        if M[i, s] != 0:
            k = M[i, s] // M[s, s]
            M, L = add_to_row(M, i, s, -k), add_to_row(L, i, s, -k)

    if np.nonzero(M[(s + 1):, s])[0].size == 0:
        if M[s, s] < 0:
            M, L = change_sign_row(M, s), change_sign_row(L, s)
        return _hnf(M, L, s + 1)
    else:
        return _hnf(M, L, s)


def row_style_hermite_normal_form(M):
    """
    calculate row-style Hermite normal form
    H = LM

    Parameters
    ----------
    M: array, (m, n)

    Returns
    -------
    H: array, (m, n)
        Hermite normal form of M, upper-triangular integer matrix
    L: array, (m, m)
        unimodular matrix s.t. H = np.dot(L, M)
    """
    MM = np.copy(M)
    L = np.eye(M.shape[0], dtype=int)
    H, L = _hnf(MM, L, s=0)
    return H, L


def column_style_hermite_normal_form(M):
    """
    calculate column-style Hermite normal form
    H = MR

    Parameters
    ----------
    M: array, (m, n)

    Returns
    -------
    H: array, (m, n)
        Hermite normal form of M, lower-triangular integer matrix
    R: array, (n, n)
        unimodular matrix s.t. H = np.dot(M, R)
    """
    H_T, R_T = row_style_hermite_normal_form(M.T)
    H = H_T.T
    R = R_T.T
    return H, R
