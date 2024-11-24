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
"""Unit tests for pyspqr extension module only.

Factored out because we run only this module in Valgrind.
"""

from unittest import TestCase, main
import numpy as np

class TestSuiteSparseQRExtension(TestCase):
    """Unit tests for pyspqr extension module."""

    def test_import(self):
        """Test import."""
        import _pyspqr

    def test_qr_valid(self):
        """Test on a simple matrix."""

        m = 2
        n = 3
        # a = sp.sparse.rand(2,3,.99,'csc')
        # a.data, a.indices, a.indptr
        data = np.array([0.56080895, 0.38371089, 0.10165425, 0.61134812, 0.60591158, 0.27545353])
        indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        indptr = np.array([0, 2, 4, 6], dtype=np.int32)

        from _pyspqr import qr, q_multiply

        for ordering in range(10):
            result = qr(m, n, data, indices, indptr, ordering)
            print(result)
            # run the q multiplication
            _, h_tuple, _, h_tau, _ = result
            vector = np.ones(m, dtype=float)
            _, _, hdata, hindices, hindptr = h_tuple
            # sadly this needs to be done manually
            hindptr = np.concatenate([hindptr, [len(hdata)]], dtype=np.int32)
            q_multiply(m, len(h_tau), True, vector, h_tau, hdata, hindices, hindptr)
            q_multiply(m, len(h_tau), False, vector, h_tau, hdata, hindices, hindptr)
            self.assertTrue(np.allclose(vector, np.ones(2)))

    def test_qr_inputs(self):
        "Input checking for QR function."

        m = 2
        n = 3
        # a = sp.sparse.rand(2,3,.99,'csc')
        # a.data, a.indices, a.indptr
        data = np.array([0.56080895, 0.38371089, 0.10165425, 0.61134812, 0.60591158, 0.27545353])
        indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        indptr = np.array([0, 2, 4, 6], dtype=np.int32)

        ordering = 5 # AMD

        from _pyspqr import qr

        with self.assertRaises(TypeError):
            qr(m + .1, n, data, indices, indptr, ordering)

        with self.assertRaises(TypeError):
            qr(m, 'hi', data, indices, indptr, ordering)

        with self.assertRaises(TypeError):
            qr(data)

        with self.assertRaises(TypeError):
            qr(data, indices)

        with self.assertRaises(TypeError):
            qr(m, n, data.astype(int), indices, indptr, ordering)

        with self.assertRaises(TypeError):
            qr(m, n, data, indices.astype(int), indptr, ordering)

        with self.assertRaises(TypeError):
            qr(m, n, data, indices, indptr.astype(int), ordering)

        with self.assertRaises(TypeError):
            qr(m, n, data[::2], indices, indptr, ordering)

        with self.assertRaises(TypeError):
            qr(m, n, data, indices[::2], indptr, ordering)

        with self.assertRaises(TypeError):
            qr(m, n, data, indices, indptr[::2], ordering)

        with self.assertRaises(ValueError):
            qr(m, n, data, indices[:-1], indptr, ordering)

    def test_wrong_CSC_format_inputs(self):
        "Check errors caught by SuiteSparse input validation."

        m = 2
        n = 3
        # a = sp.sparse.rand(2,3,.99,'csc')
        # a.data, a.indices, a.indptr
        data = np.array([0.56080895, 0.38371089, 0.10165425, 0.61134812, 0.60591158, 0.27545353])
        indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        indptr = np.array([0, 2, 4, 6], dtype=np.int32)

        ordering = 5 # AMD

        from _pyspqr import qr
        with self.assertRaises(ValueError):
            _indptr = np.array([0, 4, 4, 6], dtype=np.int32)
            qr(m, n, data, indices, _indptr, ordering)

        with self.assertRaises(ValueError):
            _indptr = np.array([0, 4, 2, 6], dtype=np.int32)
            qr(m, n, data, indices, _indptr, ordering)

        with self.assertRaises(ValueError):
            _indptr = np.array([0, 8, 10, 20], dtype=np.int32)
            qr(m, n, data, indices, _indptr, ordering)

        with self.assertRaises(ValueError):
            _indices = np.array([-1, 1, 0, 1, 0, 1], dtype=np.int32)
            qr(m, n, data, _indices, indptr, ordering)

        with self.assertRaises(ValueError):
            _indices = np.array([2, 1, 0, 1, 0, 1], dtype=np.int32)
            qr(m, n, data, _indices, indptr, ordering)


if __name__ == '__main__':
    main()
