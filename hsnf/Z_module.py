import numpy as np


class ZmoduleHomomorphism:
    """
    homomorphism between Z-modules

    Parameters
    ----------
    A: array, (m, n)
        matrix representation of homomorhism: Z^m -> Z^n
    basis_from: array, (m, )
        basis of Z^m
    basis_to: array, (n, )
        basis of Z^n
    """
    def __init__(self, A, basis_from, basis_to):
        self._A = A
        self._basis_from = basis_from
        self._basis_to = basis_to

    @property
    def num_row(self):
        return self._A.shape[0]

    @property
    def num_column(self):
        return self._A.shape[1]

    def _swap_from(self, axis1, axis2):
        self._basis_from[[axis1, axis2]] = self._basis_from[[axis2, axis1]]
        self._A[[axis1, axis2]] = self._A[[axis2, axis1]]

    def _swap_to(self, axis1, axis2):
        self._basis_to[:, [axis1, axis2]] = self._basis_to[:, [axis2, axis1]]
        self._A[:, [axis1, axis2]] = self._A[:, [axis2, axis1]]

    def _change_sign_from(self, axis):
        self._basis_from[axis] *= -1
        self._A[axis, :] *= -1

    def _change_sign_to(self, axis):
        self._basis_to[:, axis] *= -1
        self._A[:, axis] *= -1

    def _add_from(self, axis1, axis2, k):
        """
        add k times axis2 to axis1
        """
        self._basis_from[axis1] += self._basis_from[axis2] * k
        self._A[axis1, :] += self._A[axis2, :] * k

    def _add_to(self, axis1, axis2, k):
        """
        add k times axis2 to axis1
        """
        self._basis_to[:, axis1] += self._basis_to[:, axis2] * k
        self._A[:, axis1] += self._A[:, axis2] * k

    def _get_min_abs(self, s, row_only=False, column_only=False):
        """
        return argmin_{i, j} abs(A[i, j]) s.t. (i >= s and j >= s and A[i, j] != 0)
        """
        if (not row_only) and (not column_only):
            ret = (None, None)
            valmin = np.max(np.abs(self._A[s:, s:]))
            for i in range(s, self.num_row):
                for j in range(s, self.num_column):
                    if (self._A[i, j] != 0) and abs(self._A[i, j]) <= valmin:
                        ret = i, j
                        valmin = abs(self._A[i, j])
        elif row_only and (not column_only):
            ret = s
            valmin = np.max(np.abs(self._A[s:, s])) + 1
            for i in range(s, self.num_row):
                if (self._A[i, s] != 0) and abs(self._A[i, s]) < valmin:
                    ret = i
                    valmin = abs(self._A[i, s])
        elif (not row_only) and column_only:
            ret = s
            valmin = np.max(np.abs(self._A[s, s:])) + 1
            for i in range(s, self.num_column):
                if (self._A[s, i] != 0) and abs(self._A[s, i]) < valmin:
                    ret = i
                    valmin = abs(self._A[s, i])
        else:
            raise ValueError('invalid parameters')

        return ret

    def _is_lone(self, s):
        if np.nonzero(self._A[s, (s + 1):])[0].size != 0:
            return False
        if np.nonzero(self._A[(s + 1):, s])[0].size != 0:
            return False
        return True

    def _get_nextentry(self, s):
        """
        return entry which is not diviable by A[s, s]
        assume A[s, s] is not zero.
        """
        for i in range(s + 1, self.num_row):
            for j in range(s + 1, self.num_column):
                if self._A[i, j] % self._A[s, s] != 0:
                    return i, j
        return None

    def _snf(self, s):
        """
        determine up to the s-th row and column elements
        """
        if s == min(self._A.shape):
            return self._A, self._basis_from, self._basis_to

        # choose a pivot
        col, row = self._get_min_abs(s)
        if col is None:
            return self._A, self._basis_from, self._basis_to
        self._swap_from(s, col)
        self._swap_to(s, row)

        # eliminate the s-th column entries
        for i in range(s + 1, self.num_row):
            if self._A[i, s] != 0:
                k = self._A[i, s] // self._A[s, s]
                self._add_from(i, s, -k)

        # eliminate the s-th row entries
        for j in range(s + 1, self.num_column):
            if self._A[s, j] != 0:
                k = self._A[s, j] // self._A[s, s]
                self._add_to(j, s, -k)

        if self._is_lone(s):
            res = self._get_nextentry(s)
            if res:
                i, j = res
                self._add_from(s, i, 1)
                return self._snf(s)
            elif self._A[s, s] < 0:
                self._change_sign_from(s)
            return self._snf(s + 1)
        else:
            return self._snf(s)

    def smith_normal_form(self):
        """
        calculate Smith normal form

        Returns
        -------
        D: array, (m, n)
        L: array, (m, m)
        R: array, (n, n)
            D = np.dot(L, np.dot(M, R))
            L, R are unimodular.
        """
        A = self._A.copy()
        basis_from = self._basis_from.copy()
        basis_to = self._basis_to.copy()

        D, L, R = self._snf(s=0)

        # revert A, basis_from, and basis_to
        self._A = A
        self._basis_from = basis_from
        self._basis_to = basis_to

        return D, L, R

    def _hnf_row(self, s):
        """
        determine up to the s-th row and column elements
        """
        if s == min(self._A.shape):
            return self._A, self._basis_from

        # choose a pivot
        row = self._get_min_abs(s, row_only=True)
        if row is None:
            return self._A, self._basis_from
        self._swap_from(s, row)

        # eliminate the s-th column entries
        for i in range(s + 1, self.num_row):
            if self._A[i, s] != 0:
                k = self._A[i, s] // self._A[s, s]
                self._add_from(i, s, -k)

        if np.nonzero(self._A[(s + 1):, s])[0].size == 0:
            if self._A[s, s] < 0:
                self._change_sign_from(s)
            return self._hnf_row(s + 1)
        else:
            return self._hnf_row(s)

    def _hnf_column(self, s):
        """
        determine up to the s-th row and column elements
        """
        if s == min(self._A.shape):
            return self._A, self._basis_to

        # choose a pivot
        col = self._get_min_abs(s, column_only=True)
        if col is None:
            return self._A, self._basis_to
        self._swap_to(s, col)

        # eliminate the s-th row entries
        for j in range(s + 1, self.num_column):
            if self._A[s, j] != 0:
                k = self._A[s, j] // self._A[s, s]
                self._add_to(j, s, -k)

        if np.nonzero(self._A[s, (s + 1):])[0].size == 0:
            if self._A[s, s] < 0:
                self._change_sign_to(s)
            return self._hnf_column(s + 1)
        else:
            return self._hnf_column(s)

    def hermite_normal_form(self, style='row'):
        """
        calculate row-style Hermite normal form

        Parameters
        ----------
        style: str, 'row' or 'column'

        Returns
        -------
        if style == 'row'
            H: array, (m, n)
                Hermite normal form of M, upper-triangular integer matrix
            L: array, (m, m)
                unimodular matrix s.t. H = np.dot(L, M)

        if style == 'column'
            H: array, (m, n)
                Hermite normal form of M, lower-triangular integer matrix
            R: array, (n, n)
                unimodular matrix s.t. H = np.dot(M, R)
        """
        A = self._A.copy()
        basis_from = self._basis_from.copy()
        basis_to = self._basis_to.copy()

        if style == 'row':
            H, L = self._hnf_row(s=0)
        elif style == 'column':
            H, R = self._hnf_column(s=0)
        else:
            raise ValueError('unknown hermite normal form style')

        # revert A, basis_from, and basis_to
        self._A = A
        self._basis_from = basis_from
        self._basis_to = basis_to

        if style == 'row':
            return H, L
        elif style == 'column':
            return H, R

    @classmethod
    def _standard_basis(cls, n):
        return np.eye(n, dtype=int)

    @classmethod
    def with_standard_basis(cls, A):
        """
        Parameters
        ----------
        A: array, (m, n)
            matrix representation of homomorhism: Z^m -> Z^n
        """
        A = np.array(A, dtype=int)
        if A.ndim != 2:
            raise ValueError("matrix representation must be 2d")

        m, n = A.shape
        basis_from = cls._standard_basis(m)
        basis_to = cls._standard_basis(n)

        return cls(A, basis_from, basis_to)


def smith_normal_form(M):
    """
    calculate Smith normal form

    Parameters
    ----------
    M: array, (m, n)
        integer matrix

    Returns
    -------
    D: array, (m, n)
    L: array, (m, m)
    R: array, (n, n)
        D = np.dot(L, np.dot(M, R))
        L, R are unimodular.
    """
    zmh = ZmoduleHomomorphism.with_standard_basis(M)
    return zmh.smith_normal_form()


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
    zmh = ZmoduleHomomorphism.with_standard_basis(M)
    return zmh.hermite_normal_form(style='row')


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
    zmh = ZmoduleHomomorphism.with_standard_basis(M)
    return zmh.hermite_normal_form(style='column')
