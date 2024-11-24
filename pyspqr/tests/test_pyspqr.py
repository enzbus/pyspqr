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

# We use numpy.isclose for accuracy check
ABS_ACCURACY = 1e-13
REL_ACCURACY = 1e-13

# We also compare accuracy vs dense QR Numpy/LAPACK accuracy
ACCURACY_LOSS_DENSE = 4.

class TestSuiteSparseQR(TestCase):
    """Unit tests for pyspqr."""

    def _check_fwd_mult(self, A, Q, R, E, Adense, Qdense, Rdense):
        """Check forward multiplication."""
        x = np.random.randn(A.shape[1])
        aprod = A @ x
        qrprod = Q @ (R @ (E @ x))

        adenseprod = Adense @ x
        qrdenseprod = Qdense @ (Rdense @ x)

        self.assertLessEqual(
            np.linalg.norm(aprod - qrprod),
            ACCURACY_LOSS_DENSE * np.linalg.norm(adenseprod - qrdenseprod))

        self.assertLessEqual(
            np.max(np.abs(aprod - qrprod)),
            ACCURACY_LOSS_DENSE * np.max(np.abs(adenseprod - qrdenseprod)))

        self.assertTrue(
            np.allclose(aprod, qrprod, atol=ABS_ACCURACY, rtol=REL_ACCURACY))

    def _check_bwd_mult(self, A, Q, R, E, Adense, Qdense, Rdense):
        """Check backward multiplication."""
        x = np.random.randn(A.shape[0])
        atprod = A.T @ x
        qrtprod = E.T @ (R.T @ (Q.T @ x))

        atdenseprod = Adense.T @ x
        qrtdenseprod = Rdense.T @ (Qdense.T @ x)

        self.assertLessEqual(
            np.linalg.norm(atprod - qrtprod),
            ACCURACY_LOSS_DENSE * np.linalg.norm(atdenseprod - qrtdenseprod))

        self.assertLessEqual(
            np.max(np.abs(atprod - qrtprod)),
            ACCURACY_LOSS_DENSE * np.max(np.abs(atdenseprod - qrtdenseprod)))

        self.assertTrue(
            np.allclose(atprod, qrtprod, atol=ABS_ACCURACY, rtol=REL_ACCURACY))

    def _qr_check(self, A):
        """Base test for given matrix."""
        Adense = A.todense().A
        Qdense, Rdense = np.linalg.qr(Adense)
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
                self._check_fwd_mult(A, Q, R, E, Adense, Qdense, Rdense)
                self._check_bwd_mult(A, Q, R, E, Adense, Qdense, Rdense)

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
