# Copyright (C) 2024 Enzo Busseti
#
# This file is part of Pyspqr.
#
# Pyspqr is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Pyspqr is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Pyspqr. If not, see <https://www.gnu.org/licenses/>.
"""Python bindings for SuiteSparseQR."""

from __future__ import annotations

import numpy as np
import scipy as sp
from _pyspqr import qr as _qr

__all__ = ['qr', 'HouseholderOrthogonal', 'Permutation']


class HouseholderOrthogonal(sp.sparse.linalg.LinearOperator):
    """Orthogonal linear operator with Householder reflections."""

    def _rmatvec(self, input_vector):
        result = np.empty_like(input_vector)
        result[self.permutation] = input_vector
        for k in range(self.n_reflections):
            col = self.householder_reflections[:, k].todense().A1
            result -= ((col @ result) * self.householder_coefficients[k]) * col
        return result

    def _matvec(self, input_vector):
        result = np.array(input_vector, copy=True)
        for k in range(self.n_reflections)[::-1]:
            col = self.householder_reflections[:, k].todense().A1
            result -= ((col @ result) * self.householder_coefficients[k]) * col
        return result[self.permutation]

    def __init__(
        self,
        householder_reflections: sp.sparse.csc_matrix,
        householder_coefficients: np.array,
        permutation: np.array,
        ):

        self.householder_reflections = householder_reflections
        self.householder_coefficients = householder_coefficients
        self.permutation = permutation
        self.n_reflections = self.householder_reflections.shape[1]

        m = len(self.permutation)
        super().__init__(dtype=float, shape=(m, m))


class Permutation(sp.sparse.linalg.LinearOperator):
    """Permutation linear operator."""

    def _matvec(self, vector):
        return vector[self.permutation]

    def _rmatvec(self, vector):
        result = np.empty_like(vector)
        result[self.permutation] = vector
        return result

    def __init__(self, permutation):
        self.permutation = permutation
        n = len(permutation)
        super().__init__(shape=(n, n), dtype=float)


def _make_csc_matrix(m, n, data, indices, indptr):
    """Convert matrix returned by SuiteSparse to Scipy CSC matrix.

    There are a few caveats, and corner cases (e.g., empty matrix) need special
    treatment.
    """

    # empty matrix, no rows or no columns; OR data = [0.]; latter is important
    if (n == 0) or (m == 0) or ((len(data) == 1) and (data[0] == 0.)):
        return sp.sparse.csc_matrix((m, n), dtype=float)

    # in non-empty case, SuiteSparse doesn't store the last element of indptr,
    # which Scipy uses
    if len(indptr) != n+1:
        indptr = np.concatenate([indptr, [len(data)]], dtype=np.int32)

    return sp.sparse.csc_matrix((data, indices, indptr), shape=(m, n))

def qr(matrix: sp.sparse.csc_matrix):
    """Factorize Scipy sparse CSC matrix."""
    matrix = sp.sparse.csc_matrix(matrix)
    r_tuple, h_tuple, h_pinv, h_tau, e = _qr(
        matrix.shape[0], matrix.shape[1], matrix.data, matrix.indices,
        matrix.indptr)
    r_csc = _make_csc_matrix(*r_tuple)
    h_csc = _make_csc_matrix(*h_tuple)
    q = HouseholderOrthogonal(h_csc, h_tau, h_pinv)
    return q, r_csc, Permutation(e)
