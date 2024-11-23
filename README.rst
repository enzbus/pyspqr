
``pyspqr``: Simple Python Wrapper for `SuiteSparseQR <https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/SPQR>`__
==========================================================================================================================

.. code-block:: python

    import scipy as sp
    from pyspqr import qr

    A = sp.sparse.random(1000, 1000, format='csc')

    Q, R, E = qr(A)

Where `Q` is an orthogonal linear operator represented by Householder
reflections plus a permutation, `R` is a sparse upper triangular matrix,
and `E` is a permutation linear operator. They are used as follows:

.. code-block:: python

    import numpy as np
    from scipy.sparse.linalg import spsolve, spsolve_triangular

    x = np.random.randn(1000)

    assert np.allclose(Q @ (R @ (E @ x)), A @ x)

    assert np.allclose(
        E.T @ spsolve_triangular(R, Q.T @ x, lower=False),
        spsolve(A, x)
        )

We are working on offering an easy interface when the input matrix is not
square and/or not full rank, and the user is interested in a least-squares
or minimum-norm solution. (It's already doable also in those cases with the
current outputs, but it's not too user-friendly.)

Installation
============

.. code-block:: bash

    pip install pyspqr

We publish compiled wheels in PyPI for all the major platforms, including
Windows and ARM64 Linux. `See here <https://pypi.org/project/pyspqr/#files>`__
for all platforms and architectures available. They "just work".

We use ABI3 reduced Python3 API and the newer Numpy2 ABI, so the wheels run on
any Python greater or equal than 3.6, both with Numpy 1 and 2.

Wheels ship bundled with the latest version of SuiteSparse, compiled in CI/CD
by this repository, and also include an optimized multi-threaded OpenBLAS,
which is used by SuiteSparse for supernodal operations.
