CCFLAGS = -fPIC
PYTHON_INCLUDE = $(shell pkg-config --cflags --libs python3)
NUMPY_INCLUDE = -I$(shell python -c "import numpy; print(numpy.get_include())")
SPQR_INCLUDE = $(shell pkg-config --cflags --libs SPQR)
CHOLMOD_INCLUDE = $(shell pkg-config --cflags --libs CHOLMOD)
# gh runner uses ancient debian package without pkg-config stubs
# FALLBACK = -I/usr/include/suitesparse/ -lcholmod -lspqr

# Valgrind: Numpy causes some "possibly lost" errors, check with:
# valgrind --leak-check=yes python -c "import numpy"
# so we suppress those errors;
VALGRIND_FLAGS = --leak-check=yes --errors-for-leak-kinds=definite --error-exitcode=1

default: test

_suitesparseqr.so: _suitesparseqr.c
	gcc _suitesparseqr.c -shared -o _suitesparseqr.so $(CCFLAGS) $(PYTHON_INCLUDE) $(NUMPY_INCLUDE) $(SPQR_INCLUDE) $(CHOLMOD_INCLUDE)

test: _suitesparseqr.so
	python test.py

valgrind: CCFLAGS += -g -O0
valgrind: clean _suitesparseqr.so
	valgrind $(VALGRIND_FLAGS) python -c "import _suitesparseqr" 
	valgrind $(VALGRIND_FLAGS) python test.py

build: env clean
	env/bin/pip install -e .

env:
	python -m venv --system-site-packages env

clean:
	rm *.so || true
	rm -rf build || true
	rm -rf *.egg-info | true