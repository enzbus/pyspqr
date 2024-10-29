from unittest import TestCase, main

import numpy as np

class TestSuiteSparseQR(TestCase):
    """Unit tests for suitesparseqr."""

    def test_import(self):
        """Test import."""
        import suitesparseqr

    def test_qr_inputs(self):
        "Input checking for QR function."

        data = np.zeros(5, dtype=float)
        indices = np.zeros(5, dtype=np.int32)
        indptr = np.zeros(3, dtype=np.int32)

        from suitesparseqr import qr
        qr(3, 2, data, indices, indptr)

        with self.assertRaises(TypeError):
            qr(3.1,2, data, indices, indptr)

        with self.assertRaises(TypeError):
            qr(3, 'hi', data, indices, indptr)
        
        with self.assertRaises(TypeError):
            qr(data)

        with self.assertRaises(TypeError):
            qr(data, indices)
        
        with self.assertRaises(TypeError):
            qr(3,2, data.astype(int), indices, indptr)    

        with self.assertRaises(TypeError):
            qr(3,2, data, indices.astype(int), indptr)   

        with self.assertRaises(TypeError):
            qr(3,2, data, indices, indptr.astype(int))   

        with self.assertRaises(TypeError):
            qr(3,2, data[::2], indices, indptr)

        with self.assertRaises(TypeError):
            qr(3,2, data, indices[::2], indptr)

        with self.assertRaises(TypeError):
            qr(3,2, data, indices, indptr[::2])

        with self.assertRaises(ValueError):
            qr(3,2, data, indices[:-1], indptr)  

if __name__ == '__main__':
    main()