from unittest import TestCase, main

import numpy as np

class TestSuiteSparseQR(TestCase):
    """Unit tests for suitesparseqr."""

    def test_import(self):
        """Test import."""
        import _suitesparseqr

    def test_qr_inputs(self):
        "Input checking for QR function."

        m = 2
        n = 3
        # a = sp.sparse.rand(2,3,.99,'csc')
        # a.data, a.indices, a.indptr
        data = np.array([0.56080895, 0.38371089, 0.10165425, 0.61134812, 0.60591158, 0.27545353])
        indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        indptr = np.array([0, 2, 4, 6], dtype=np.int32)

        from _suitesparseqr import qr
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
            qr(m,n, data.astype(int), indices, indptr)    

        with self.assertRaises(TypeError):
            qr(m,n, data, indices.astype(int), indptr)   

        with self.assertRaises(TypeError):
            qr(m,n, data, indices, indptr.astype(int))   

        with self.assertRaises(TypeError):
            qr(m,n, data[::2], indices, indptr)

        with self.assertRaises(TypeError):
            qr(m,n, data, indices[::2], indptr)

        with self.assertRaises(TypeError):
            qr(m,n, data, indices, indptr[::2])

        with self.assertRaises(ValueError):
            qr(m,n, data, indices[:-1], indptr)

    def test_wrong_CSC_format_inputs(self):
        "Check errors caught by SuiteSparse input validation."

        m = 2
        n = 3
        # a = sp.sparse.rand(2,3,.99,'csc')
        # a.data, a.indices, a.indptr
        data = np.array([0.56080895, 0.38371089, 0.10165425, 0.61134812, 0.60591158, 0.27545353])
        indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        indptr = np.array([0, 2, 4, 6], dtype=np.int32)

        from _suitesparseqr import qr
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