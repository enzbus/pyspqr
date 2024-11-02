
``pyspqr``: Simple Python Wrapper for `SuiteSparseQR <https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/SPQR>`__
==========================================================================================================================

.. code-block:: python

    import scipy as sp
    from pyspqr import qr
    
    A = sp.sparse.random(1000,1000, format='csc')

    R, H, HPinv, HTau = qr(A)


The result objects are Scipy CSC sparse matrices or 1 dimensional Numpy arrays.
The last three objects are the Householder reflection representing Q, plus a row
permutation. In future versions we'll wrap them in a ``scipy.sparse.LinearOperator``

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
