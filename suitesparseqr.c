#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <SuiteSparseQR_C.h>

static PyObject *qr(PyObject *self, PyObject *args){
    /* We use names of Scipy sparse CSC.*/
    PyArrayObject *data_np;
    PyArrayObject *indices_np;
    PyArrayObject *indptr_np;
    
    PyArg_ParseTuple(args, "OOO", &data_np, &indices_np, &indptr_np);
    
    if (PyErr_Occurred()){
        return NULL;
    }
    if (!PyArray_Check(data_np) || PyArray_TYPE(data_np) != NPY_DOUBLE || !PyArray_IS_C_CONTIGUOUS(data_np)){
        PyErr_SetString(PyExc_TypeError,
           "First argument must be contiguous double Numpy array.");
        return NULL;
    }
    double * data_arr = PyArray_DATA(data_np);

    if (!PyArray_Check(indices_np) || PyArray_TYPE(indices_np) != NPY_INT32 || !PyArray_IS_C_CONTIGUOUS(indices_np)){
        PyErr_SetString(PyExc_TypeError,
            "Second argument must be contiguous int32 Numpy array.");
        return NULL;
    }
    int32_t * indices_arr = PyArray_DATA(indices_np);

    if (!PyArray_Check(indptr_np) || PyArray_TYPE(indptr_np) != NPY_INT32 || !PyArray_IS_C_CONTIGUOUS(indptr_np)){
        PyErr_SetString(PyExc_TypeError,
            "Third argument must be contiguous int32 Numpy array.");
        return NULL;
    }
    int32_t * indptr_arr = PyArray_DATA(indptr_np);

    return PyFloat_FromDouble(0.);
};

static PyMethodDef methods[] = {
    {"qr", qr, METH_VARARGS, "Perform sparse QR decomposition."},
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
    import_array();
    return PyModuleDef_Init(&suitesparseqr);
}

