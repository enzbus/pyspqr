from unittest import TestCase, main

import numpy as np

class TestSuiteSparseQR(TestCase):
    """Unit tests for suitesparseqr."""

    def test_import(self):
        """Test import."""
        import suitesparseqr

    def test_qr_basic(self):
        data = np.zeros(5, dtype=float)
        indices = np.zeros(5, dtype=np.int32)
        indptr = np.zeros(2, dtype=np.int32)
        from suitesparseqr import qr
        qr(data, indices, indptr)
        
        with self.assertRaises(TypeError):
            qr(data)

        with self.assertRaises(TypeError):
            qr(data, indices)
        
        with self.assertRaises(TypeError):
            qr(data.astype(int), indices, indptr)    

        with self.assertRaises(TypeError):
            qr(data, indices.astype(int), indptr)   

        with self.assertRaises(TypeError):
            qr(data, indices, indptr.astype(int))   

        with self.assertRaises(TypeError):
            qr(data[::2], indices, indptr)

        with self.assertRaises(TypeError):
            qr(data, indices[::2], indptr)   

if __name__ == '__main__':
    main()