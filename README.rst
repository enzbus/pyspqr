``pyspqr``: Simple Python Wrapper for SuiteSparseQR
===================================================

.. code-block:: python

    import scipy as sp
    from pyspqr import qr
    
    A = sp.sparse.random(1000,1000, format='csc')

    R, H, HPinv, HTau = qr(A)


The result objects are Scipy CSC sparse matrices or 1 dimensional Numpy arrays.
The last three objects are the Householder reflection representing Q, plus a row
permutation. In future versions we'll wrap them in a ``scipy.sparse.LinearOperator``

