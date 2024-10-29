#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <SuiteSparseQR_C.h>

static PyObject *qr(PyObject *self, PyObject *args){
    /* We use names of Scipy sparse CSC.*/
    int m;
    int n;
    PyArrayObject *data_np;
    PyArrayObject *indices_np;
    PyArrayObject *indptr_np;
    
    PyArg_ParseTuple(args, "iiOOO", &m, &n, &data_np, &indices_np, &indptr_np);
    
    if (PyErr_Occurred()){
        return NULL;
    }
    if (!PyArray_Check(data_np) || PyArray_TYPE(data_np) != NPY_DOUBLE || !PyArray_IS_C_CONTIGUOUS(data_np)){
        PyErr_SetString(PyExc_TypeError,
           "Data argument must be contiguous double Numpy array.");
        return NULL;
    }
    double * data_arr = PyArray_DATA(data_np);

    if (!PyArray_Check(indices_np) || PyArray_TYPE(indices_np) != NPY_INT32 || !PyArray_IS_C_CONTIGUOUS(indices_np)){
        PyErr_SetString(PyExc_TypeError,
            "Indices argument must be contiguous int32 Numpy array.");
        return NULL;
    }
    int32_t * indices_arr = PyArray_DATA(indices_np);

    if (!PyArray_Check(indptr_np) || PyArray_TYPE(indptr_np) != NPY_INT32 || !PyArray_IS_C_CONTIGUOUS(indptr_np)){
        PyErr_SetString(PyExc_TypeError,
            "Indptr argument must be contiguous int32 Numpy array.");
        return NULL;
    }
    int32_t * indptr_arr = PyArray_DATA(indptr_np);

    size_t nnz = PyArray_SIZE(data_np);
    if (nnz != PyArray_SIZE(indices_np)){
        PyErr_SetString(PyExc_ValueError,
            "Data and indices arrays must have the same length.");
        return NULL;
    }

    if (n+1 != PyArray_SIZE(indptr_np)){
        PyErr_SetString(PyExc_ValueError,
            "Indptr array must have have length n+1.");
        return NULL;
    }

    /*SuiteSparse QR.*/
    cholmod_common Common, *cc;
    cc = &Common;
    if (!cholmod_start(cc)){
        return NULL;
    }


    cholmod_sparse * input_matrix = cholmod_allocate_sparse(
        m, //size_t nrow,    // # of rows
        n, //size_t ncol,    // # of columns
        nnz, //size_t nzmax,   // max # of entries the matrix can hold
        true, //int sorted,     // true if columns are sorted
        true, //int packed,     // true if A is be packed (A->nz NULL), false if unpacked
        0, //int stype,      // the stype of the matrix (unsym, tril, or triu)
        CHOLMOD_DOUBLE+CHOLMOD_REAL,
        cc);

    if (!input_matrix){
        return NULL;
    }

    if (!cholmod_print_sparse(input_matrix, "input matrix", cc)){
        return NULL;
    }

    if (!cholmod_finish(cc)){
        return NULL;
    }

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

