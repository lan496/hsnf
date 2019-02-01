import numpy as np


# http://blog.dlfer.xyz/post/2016-10-27-smith-normal-form/
def swap_rows(M, i, j):
    ret = M.copy()
    ret[i, :], ret[j, :] = M[j, :], M[i, :]
    return ret


def swap_columns(M, i, j):
    ret = M.copy()
    ret[:, i], ret[:, j] = M[:, j], M[:, i]
    return ret


def add_to_row(M, i, j, k):
    ret = M.copy()
    ret[i, :] += M[j, :] * k
    return ret


def add_to_column(M, i, j, k):
    ret = M.copy()
    ret[:, i] += M[:, j] * k
    return ret


def change_sign_row(M, i):
    ret = M.copy()
    ret[i, :] *= -1
    return ret


def change_sign_column(M, i):
    ret = M.copy()
    ret[:, i] *= -1
    return ret


def get_min_abs(M, s):
    """
    return argmin_{i, j} abs(M[i, j]) s.t. (i >= s and j >= s and M[i, j] != 0)
    """
    ret = (None, None)
    valmin = np.max(np.abs(M[s:, s:]))
    for i in range(s, M.shape[0]):
        for j in range(s, M.shape[1]):
            if (M[i, j] != 0) and abs(M[i, j]) <= valmin:
                ret = i, j
                valmin = abs(M[i, j])
    return ret


def is_lone(M, s):
    if np.nonzero(M[s, (s + 1):])[0].size != 0:
        return False
    if np.nonzero(M[(s + 1):, s])[0].size != 0:
        return False
    return True


def get_nextentry(M, s):
    """
    return entry which is not diviable by M[s, s]
    assume M[s, s] is not zero.
    """
    for i in range(s + 1, M.shape[0]):
        for j in range(s + 1, M.shape[1]):
            if M[i, j] % M[s, s] != 0:
                return i, j
    return None


def _snf(M, L, R, s):
    """
    determine up to the s-th row and column elements
    """
    if s == min(M.shape):
        return M, L, R

    # choose a pivot
    num_row, num_column = M.shape
    col, row = get_min_abs(M, s)
    if col is None:
        return M, L, R
    M, L = swap_rows(M, s, col), swap_rows(L, s, col)
    M, R = swap_columns(M, s, row), swap_columns(R, s, row)

    # eliminate the s-th column entries
    for i in range(s + 1, num_row):
        if M[i, s] != 0:
            k = M[i, s] // M[s, s]
            M, L = add_to_row(M, i, s, -k), add_to_row(L, i, s, -k)

    # eliminate the s-th row entries
    for j in range(s + 1, num_column):
        if M[s, j] != 0:
            k = M[s, j] // M[s, s]
            M, R = add_to_column(M, j, s, -k), add_to_column(R, j, s, -k)

    if is_lone(M, s):
        res = get_nextentry(M, s)
        if res:
            i, j = res
            M, L = add_to_row(M, s, i, 1), add_to_row(L, s, i, 1)
            return _snf(M, L, R, s)
        elif M[s, s] < 0:
            M, L = change_sign_row(M, s), change_sign_row(L, s)
        return _snf(M, L, R, s + 1)
    else:
        return _snf(M, L, R, s)


def smith_normal_form(M):
    """
    calculate Smith normal form

    Parameters
    ----------
    M: array, (dim, dim)

    Returns
    -------
    D: array, (dim)
    L: array, (dim)
    R: array, (dim)
        D = np.dot(L, np.dot(M, R))
        L, R are unimodular.
    """
    MM = np.copy(M)
    L = np.eye(M.shape[0], dtype=int)
    R = np.eye(M.shape[1], dtype=int)
    D, L, R = _snf(MM, L, R, s=0)
    return D, L, R
