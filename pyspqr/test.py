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
"""Unit tests for pyspqr."""
from unittest import TestCase, main
import scipy as sp
import numpy as np
from pyspqr import qr
from .test_extension import TestSuiteSparseQRExtension


class TestSuiteSparseQR(TestCase):
    """Unit tests for pyspqr."""

    def test_basic(self):
        """Test basic interface."""
        
        A = sp.sparse.random(100,100, density=1., format='csc')
        qr(A)
        A = sp.sparse.random(100,20, density=1., format='csc')
        qr(A)
        A = sp.sparse.random(20,100, density=1., format='csc')
        qr(A)
        A = sp.sparse.random(1000,1000, format='csc')
        qr(A)
        A = sp.sparse.random(1000,200, format='csc')
        qr(A)
        A = sp.sparse.random(1000,200, format='csc')
        qr(A)

if __name__ == '__main__':
    main()