PYTHON_INCLUDE = $(shell pkg-config --cflags --libs python3)
NUMPY_INCLUDE = -I$(shell python -c "import numpy; print(numpy.get_include())")

default: test

suitesparseqr.so: suitesparseqr.c
	gcc suitesparseqr.c -shared -o suitesparseqr.so $(PYTHON_INCLUDE) $(NUMPY_INCLUDE) -fPIC

test: suitesparseqr.so
	python test.py

clean:
	rm *.so