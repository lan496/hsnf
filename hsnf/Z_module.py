# Copyright (c) 2019 Kohei Shinohara
# Distributed under the terms of the MIT License.
from __future__ import annotations

import numpy as np

from hsnf.utils import NDArrayInt, get_nonzero_min_abs_full, get_nonzero_min_abs_row


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

    def _is_lone(self, s):
        """
        check if all s-th row elements column elements become zero
        """
        if np.nonzero(self._A[s, (s + 1) :])[0].size != 0:
            return False
        if np.nonzero(self._A[(s + 1) :, s])[0].size != 0:
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
        determine SNF up to the s-th row and column elements
        """
        if s == min(self._A.shape):
            return self._A, self._basis_from, self._basis_to

        # choose a pivot
        row, col = get_nonzero_min_abs_full(self._A, s)
        if col is None:
            # if there does not remain non-zero elements, this procesure ends.
            return self._A, self._basis_from, self._basis_to
        self._swap_from(s, row)
        self._swap_to(s, col)

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

        # if there does not remain non-zero element in s-th row and column, find a next entry
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

        see the following awesome post for a description of this algorithm:
            http://www.dlfer.xyz/post/2016-10-27-smith-normal-form/

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

    def _hnf_row(self, si, sj):
        """
        determine row-style HNF up to the si-th row and the sj-th column elements
        """
        if (si == self.num_row) or (sj == self.num_column):
            return self._A, self._basis_from

        # choose a pivot
        row, _ = get_nonzero_min_abs_row(self._A, si, sj)

        if row is None:
            # if there does not remain non-zero elements, go to a next column
            return self._hnf_row(si, sj + 1)
        self._swap_from(si, row)

        # eliminate the s-th column entries
        for i in range(si + 1, self.num_row):
            if self._A[i, sj] != 0:
                k = self._A[i, sj] // self._A[si, sj]
                self._add_from(i, si, -k)

        # if there does not remain non-zero element in s-th row, find a next entry
        if np.count_nonzero(self._A[(si + 1) :, sj]) == 0:
            if self._A[si, sj] < 0:
                self._change_sign_from(si)

            if self._A[si, sj] != 0:
                for i in range(si):
                    k = self._A[i, sj] // self._A[si, sj]
                    if k != 0:
                        self._add_from(i, si, -k)

            return self._hnf_row(si + 1, sj + 1)
        else:
            return self._hnf_row(si, sj)

    def hermite_normal_form(self):
        """
        calculate row-style Hermite normal form

        Returns
        -------
        H: array, (m, n)
            Hermite normal form of M, upper-triangular integer matrix
        L: array, (m, m)
            unimodular matrix s.t. H = np.dot(L, M)
        """
        A = self._A.copy()
        basis_from = self._basis_from.copy()
        basis_to = self._basis_to.copy()

        H, L = self._hnf_row(si=0, sj=0)

        # revert A, basis_from, and basis_to
        self._A = A
        self._basis_from = basis_from
        self._basis_to = basis_to

        return H, L

    @classmethod
    def _standard_basis(cls, n):
        return np.eye(n, dtype=int)

    @classmethod
    def with_standard_basis(cls, A):
        """
        create homomorhism with regard A as a matrix representation with standard basis

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


def smith_normal_form(M: NDArrayInt) -> tuple[NDArrayInt, NDArrayInt, NDArrayInt]:
    """
    Calculate Smith normal form of integer matrix `M`.
    Returned matrices `(D, L, R)` satisfy ``D = np.dot(L, np.dot(M, R))``.

    Parameters
    ----------
    M: array, (m, n)
        Integer matrix

    Returns
    -------
    D: array, (m, n)
        Smith normal form of `M`
    L: array, (m, m)
        Unimodular matrix
    R: array, (n, n)
        Unimodular matrix
    """
    zmh = ZmoduleHomomorphism.with_standard_basis(M)
    return zmh.smith_normal_form()


def row_style_hermite_normal_form(M: NDArrayInt) -> tuple[NDArrayInt, NDArrayInt]:
    """
    Calculate row-style Hermite normal form of `M`.
    Returned matrices `(H, L)` satisfy ``H = np.dot(L, M)``.

    Parameters
    ----------
    M: array, (m, n)
        Integer matrix

    Returns
    -------
    H: array, (m, n)
        Hermite normal form of M, upper-triangular integer matrix
    L: array, (m, m)
        Unimodular matrix
    """
    zmh = ZmoduleHomomorphism.with_standard_basis(M)
    return zmh.hermite_normal_form()


def column_style_hermite_normal_form(M: NDArrayInt) -> tuple[NDArrayInt, NDArrayInt]:
    """
    Calculate column-style Hermite normal form of `M`
    Returned matrices `(H, R)` satisfy ``H = np.dot(M, R)``

    Parameters
    ----------
    M: array, (m, n)
        Integer matrix

    Returns
    -------
    H: array, (m, n)
        Hermite normal form of M, lower-triangular integer matrix
    R: array, (n, n)
        Unimodular matrix
    """
    zmh = ZmoduleHomomorphism.with_standard_basis(M.T)
    H_T, R_T = zmh.hermite_normal_form()
    H = H_T.T
    R = R_T.T
    return H, R
