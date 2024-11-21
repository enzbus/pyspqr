
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

We publish compiled wheels in PyPI for all the major platforms, including aarch64 Linux.
`See here <https://pypi.org/project/pyspqr/#files>`__ for all platforms and
architectures available.

We use ABI3 reduced Python3 API and the newer Numpy2 ABI, so the wheels run on
any Python greater or equal than 3.6 and both on Numpy 1 and 2.

Wheels ship bundled with the latest version of SuiteSparse which we compile
ourselves in CI/CD, for Linux and Mac. They are linked to openBLAS on Linux,
Intel MKL on Windows, and Accelerate on Mac. OpenMP is enabled on Linux builds,
Windows, and Mac aarch64 10.14 or greater. We also have another build for
Mac aarch64 with more retro compatibility but without openMP.

All our packaging code is standard setuptools, with minimal tweaks (we use
``pkg-config``), so you should be able to compile locally using our source
distribution, if for example you want to link another BLAS implementation, or
use SuiteSparse CUDA kernels. The pre-built wheels should be OK for most users.
