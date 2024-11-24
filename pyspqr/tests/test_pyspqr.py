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
"""Unit tests for pyspqr."""
from unittest import TestCase
import scipy as sp
import numpy as np
from pyspqr import qr

# we use exact equality rounding to these many digits
N_DIGITS_TEST = 8

class TestSuiteSparseQR(TestCase):
    """Unit tests for pyspqr."""

    def _check_fwd_mult(self, A, Q, R, E):
        """Check forward multiplication."""
        x = np.random.randn(A.shape[1])
        self.assertListEqual(
            list(np.round(A @ x, N_DIGITS_TEST)),
            list(np.round(Q @ (R @ (E @ x)), N_DIGITS_TEST)))

    def _check_bwd_mult(self, A, Q, R, E):
        """Check backward multiplication."""
        x = np.random.randn(A.shape[0])
        self.assertListEqual(
            list(np.round(A.T @ x, N_DIGITS_TEST)),
            list(np.round(E.T @ (R.T @ (Q.T @ x)), N_DIGITS_TEST)))

    def _qr_check(self, A):
        """Base test for given matrix."""
        for ordering in [
            'FIXED',
            'NATURAL',
            'COLAMD',
            'GIVEN',
            'CHOLMOD',
            'AMD',
            'METIS',
            'DEFAULT',
            'BEST',
            'BESTAMD',
        ]:
            with self.subTest(ordering=ordering):
                Q, R, E = qr(A, ordering)
                self._check_fwd_mult(A, Q, R, E)
                self._check_bwd_mult(A, Q, R, E)

    def test_corner(self):
        """Test with some corner cases."""

        # empty matrix
        for m, n in ((10, 10), (10, 5), (5, 10)):
            A = sp.sparse.csc_matrix((m, n))
            self._qr_check(A)

        # super few entries matrix
        for m, n in ((10, 10), (10, 20), (20, 10)):
            A = sp.sparse.random(m, n, density=0.01, format='csc')
            self._qr_check(A)

    def test_dense(self):
        """Test with dense matrices."""

        for m, n in [(100, 100), (100, 50), (50, 100)]:

            np.random.seed(0)
            A = sp.sparse.random(m, n, density=1., format='csc')
            self._qr_check(A)

    def test_big_sparse(self):
        """Test with random big sparse."""

        for m, n in [(1000, 1000), (1000, 200), (200, 1000)]:
            np.random.seed(0)
            A = sp.sparse.random(m, n, density = 0.01, format='csc')
            self._qr_check(A)