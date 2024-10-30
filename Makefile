CCFLAGS = -fPIC
PYTHON_INCLUDE = $(shell pkg-config --cflags --libs python3)
NUMPY_INCLUDE = -I$(shell python -c "import numpy; print(numpy.get_include())")
SPQR_INCLUDE = $(shell pkg-config --cflags --libs SPQR)
CHOLMOD_INCLUDE = $(shell pkg-config --cflags --libs CHOLMOD)
# gh runner uses ancient debian package without pkg-config stubs
# FALLBACK = -I/usr/include/suitesparse/ -lcholmod -lspqr

default: test

suitesparseqr.so: suitesparseqr.c
	gcc suitesparseqr.c -shared -o suitesparseqr.so $(CCFLAGS) $(PYTHON_INCLUDE) $(NUMPY_INCLUDE) $(SPQR_INCLUDE) $(CHOLMOD_INCLUDE)

test: suitesparseqr.so
	python test.py

valgrind: CCFLAGS += -g -O0
valgrind: clean suitesparseqr.so
	valgrind --leak-check=yes python test.py

clean:
	rm *.so || true