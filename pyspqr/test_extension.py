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

    def test_qr_inputs(self):
        "Input checking for QR function."

        m = 2
        n = 3
        # a = sp.sparse.rand(2,3,.99,'csc')
        # a.data, a.indices, a.indptr
        data = np.array([0.56080895, 0.38371089, 0.10165425, 0.61134812, 0.60591158, 0.27545353])
        indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        indptr = np.array([0, 2, 4, 6], dtype=np.int32)

        from _pyspqr import qr
        result = qr(m, n, data, indices, indptr)
        print(result)

        with self.assertRaises(TypeError):
            qr(m + .1, n, data, indices, indptr)

        with self.assertRaises(TypeError):
            qr(m, 'hi', data, indices, indptr)

        with self.assertRaises(TypeError):
            qr(data)

        with self.assertRaises(TypeError):
            qr(data, indices)

        with self.assertRaises(TypeError):
            qr(m, n, data.astype(int), indices, indptr)

        with self.assertRaises(TypeError):
            qr(m, n, data, indices.astype(int), indptr)

        with self.assertRaises(TypeError):
            qr(m, n, data, indices, indptr.astype(int))

        with self.assertRaises(TypeError):
            qr(m, n, data[::2], indices, indptr)

        with self.assertRaises(TypeError):
            qr(m, n, data, indices[::2], indptr)

        with self.assertRaises(TypeError):
            qr(m, n, data, indices, indptr[::2])

        with self.assertRaises(ValueError):
            qr(m, n, data, indices[:-1], indptr)

    def test_wrong_CSC_format_inputs(self):
        "Check errors caught by SuiteSparse input validation."

        m = 2
        n = 3
        # a = sp.sparse.rand(2,3,.99,'csc')
        # a.data, a.indices, a.indptr
        data = np.array([0.56080895, 0.38371089, 0.10165425, 0.61134812, 0.60591158, 0.27545353])
        indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        indptr = np.array([0, 2, 4, 6], dtype=np.int32)

        from _pyspqr import qr
        with self.assertRaises(ValueError):
            _indptr = np.array([0, 4, 4, 6], dtype=np.int32)
            qr(m, n, data, indices, _indptr)

        with self.assertRaises(ValueError):
            _indptr = np.array([0, 4, 2, 6], dtype=np.int32)
            qr(m, n, data, indices, _indptr)

        with self.assertRaises(ValueError):
            _indptr = np.array([0, 8, 10, 20], dtype=np.int32)
            qr(m, n, data, indices, _indptr)

        with self.assertRaises(ValueError):
            _indices = np.array([-1, 1, 0, 1, 0, 1], dtype=np.int32)
            qr(m, n, data, _indices, indptr)

        with self.assertRaises(ValueError):
            _indices = np.array([2, 1, 0, 1, 0, 1], dtype=np.int32)
            qr(m, n, data, _indices, indptr)


if __name__ == '__main__':
    main()
