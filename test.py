from unittest import TestCase, main

import numpy as np

class TestSuiteSparseQR(TestCase):
    """Unit tests for suitesparseqr."""

    def test_import(self):
        """Test import."""
        import suitesparseqr

    def test_qr_inputs(self):
        "Input checking for QR function."

        m = 2
        n = 3
        # a = sp.sparse.rand(2,3,.99,'csc')
        # a = sp.sparse.rand(2,3,.99,'csc')
        data = np.array([0.56080895, 0.38371089, 0.10165425, 0.61134812, 0.60591158, 0.27545353])
        indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        indptr = np.array([0, 2, 4, 6], dtype=np.int32)

        from suitesparseqr import qr
        qr(m, n, data, indices, indptr)

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

if __name__ == '__main__':
    main()