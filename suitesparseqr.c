#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <SuiteSparseQR_C.h>

static PyMethodDef methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef suitesparseqr = {
    PyModuleDef_HEAD_INIT,
    "suitesparseqr",
    "Python bindings for SuiteSparseQR, internal module.",
    0,
    methods,
};

PyMODINIT_FUNC PyInit_suitesparseqr(void)
{
    return PyModuleDef_Init(&suitesparseqr);
}

