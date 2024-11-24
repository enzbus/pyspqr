CCFLAGS = -fPIC -O3 -fopenmp-simd
PYTHON_INCLUDE = $(shell pkg-config --cflags --libs python3)
NUMPY_INCLUDE = -I$(shell python -c "import numpy; print(numpy.get_include())")
SPQR_INCLUDE = $(shell pkg-config --cflags --libs SPQR)
CHOLMOD_INCLUDE = $(shell pkg-config --cflags --libs CHOLMOD)
VENV_FLAGS =
# gh runner uses ancient debian package without pkg-config stubs
# FALLBACK = -I/usr/include/suitesparse/ -lcholmod -lspqr

OS := $(shell uname)
ifeq ($(OS), Darwin) # brew packaging is problematic
    PYTHON_INCLUDE += -lpython3.13
endif

OS := $(shell uname)
ifeq ($(OS), Linux) # *only* develop on Linux to save headaches
    VENV_FLAGS += --system-site-packages
endif

# Valgrind: Numpy causes some "possibly lost" errors, check with:
# valgrind --leak-check=yes python -c "import numpy"
# so we suppress those errors; results should be taken with a grain of salt,
# inspect if in doubt, CPython and Numpy allocate a lot of things
VALGRIND_FLAGS = --leak-check=yes --errors-for-leak-kinds=definite -s

default: test

_pyspqr.so: _pyspqr.c
	gcc _pyspqr.c -shared -o _pyspqr.so $(CCFLAGS) $(PYTHON_INCLUDE) $(NUMPY_INCLUDE) $(SPQR_INCLUDE) $(CHOLMOD_INCLUDE)

test: _pyspqr.so
	python -m pyspqr.tests

valgrind: CCFLAGS += -g -O0
valgrind: clean _pyspqr.so
	valgrind $(VALGRIND_FLAGS) python -c "import _pyspqr"
	cp pyspqr/tests/test_extension.py .
	valgrind $(VALGRIND_FLAGS) python test_extension.py
	rm test_extension.py

build: env clean
	env/bin/python -m build .
	env/bin/python -m twine check dist/*.whl
	# sudo apt install patchelf
	env/bin/python -m auditwheel repair dist/*.whl --plat linux_x86_64 -w dist/
	env/bin/python -m abi3audit --strict --report dist/*.whl

env: clean
	python -m venv $(VENV_FLAGS) env
	env/bin/pip install -e .[dev]
	env/bin/python -m pyspqr.tests

clean:
	rm *.so || true
	rm -rf build || true
	rm -rf env || true
	rm -rf dist || true 
	rm -rf *.egg-info | true

release: build
	env/bin/python -m twine upload --skip-existing dist/*.tar.gz

fix:
	env/bin/python -m autopep8 -i -r pyspqr
