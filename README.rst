SuiteSparseQR Simple Python Wrapper
===================================

.. code-block:: python

    import scipy as sp
    from suitesparseqr import spqr
    
    A = sp.sparse.random(1000,1000, format='csc')

    R, H, HPinv, HTau = spqr(A)


The result objects are Scipy CSC sparse matrices or 1 dimensional Numpy arrays.
The last three objects are the Householder reflection representing Q, plus a row
permutation. In future versions we'll wrap them in a ``scipy.sparse.LinearOperator``

