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
from _pyspqr import qr as _qr, q_multiply as _q_multiply

__all__ = ['qr', 'HouseholderOrthogonal', 'Permutation']

# See https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/8ac3f515ad91ae3d0137fe98239e52d2a689eac3/SPQR/Include/SuiteSparseQR_definitions.h#L15
_ORDERINGS = (
    ('FIXED', 0),
    ('NATURAL', 1),
    ('COLAMD', 2),
    ('GIVEN', 3),
    ('CHOLMOD', 4),
    ('AMD', 5),
    ('METIS', 6),
    ('DEFAULT', 7),
    ('BEST', 8),
    ('BESTAMD', 9),
)

def _householder_multiply_python(
    vector, householder_reflections, householder_coefficients, backward=False):
    """Householder multiplication of single vector, used to test C function."""
    for i in range(len(householder_coefficients)-1, -1, -1
            ) if backward else range(len(householder_coefficients)):
        coeff = householder_coefficients[i]
        col = householder_reflections[:, i].todense().A1
        vector -= ((col @ vector) * coeff) * col

def _householder_multiply_C(
    vector, householder_reflections, householder_coefficients, backward=False):
    """Householder multiplication of single vector, see Python version."""
    _q_multiply(
        len(vector),
        len(householder_coefficients),
        backward,
        vector,
        householder_coefficients,
        householder_reflections.data,
        householder_reflections.indices,
        householder_reflections.indptr
    )

class HouseholderOrthogonal(sp.sparse.linalg.LinearOperator):
    """Orthogonal linear operator with Householder reflections."""

    def _rmatvec(self, input_vector):
        result = self.permutation_linop.T @ input_vector
        _householder_multiply_C(
            result, self.householder_reflections,
            self.householder_coefficients, backward=False)
        return result

    def _matvec(self, input_vector):
        result = np.array(input_vector, copy=True)
        _householder_multiply_C(
            result, self.householder_reflections,
            self.householder_coefficients, backward=True)
        return self.permutation_linop @ result

    def __init__(
        self,
        householder_reflections: sp.sparse.csc_matrix,
        householder_coefficients: np.array,
        permutation: np.array,
        ):

        self.householder_reflections = householder_reflections
        self.householder_coefficients = householder_coefficients
        self.permutation_array = permutation
        self.permutation_linop = Permutation(self.permutation_array)

        m = len(self.permutation_array)
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

class IdentityLinOp(sp.sparse.linalg.LinearOperator):
    """Identity linear operator."""

    def _matvec(self, vector):
        return vector

    def _rmatvec(self, vector):
        return vector

    def __init__(self, n):
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

def qr(matrix: sp.sparse.csc_matrix, ordering='AMD'):
    """Factorize Scipy sparse CSC matrix."""
    matrix = sp.sparse.csc_matrix(matrix)
    r_tuple, h_tuple, h_pinv, h_tau, e = _qr(
        matrix.shape[0], matrix.shape[1], matrix.data, matrix.indices,
        matrix.indptr, dict(_ORDERINGS)[ordering])
    r_csc = _make_csc_matrix(*r_tuple)
    h_csc = _make_csc_matrix(*h_tuple)
    q = HouseholderOrthogonal(h_csc, h_tau, h_pinv)
    return q, r_csc, IdentityLinOp(matrix.shape[1]) if e is None else Permutation(e)
