#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <SuiteSparseQR_C.h>
#include <stdbool.h>

static inline PyObject *qr(PyObject *self, PyObject *args){
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

    memcpy(input_matrix->x, data_arr, nnz*sizeof(double));
    memcpy(input_matrix->i, indices_arr, nnz*sizeof(int32_t));
    memcpy(input_matrix->p, indptr_arr, (n+1)*sizeof(int32_t));



    if (!cholmod_check_sparse(input_matrix, cc)){
        printf("Input check failed!\n");
        return NULL;
    }



    if (!cholmod_print_sparse(input_matrix, "input matrix", cc)){
        return NULL;
    }

    int32_t rank;
    cholmod_sparse *R;
    cholmod_sparse *H;
    cholmod_dense * HTau;
    cholmod_sparse *Zsparse;
    cholmod_dense *Zdense;
    int32_t * E;
    int32_t * HPinv;

    rank = SuiteSparseQR_i_C /* returns rank(A) estimate, (-1) if failure */
(
    /* inputs: */
    3, //int ordering,               /* all, except 3:given treated as 0:fixed */
    0., //double tol,                 /* columns with 2-norm <= tol treated as 0 */
    m, //int32_t econ,               /* e = max(min(m,econ),rank(A)) */
    0, //int getCTX,                 /* 0: Z=C (e-by-k), 1: Z=C', 2: Z=X (e-by-k) */
    input_matrix, //cholmod_sparse *A,          /* m-by-n sparse matrix to factorize */
    NULL, //cholmod_sparse *Bsparse,    /* sparse m-by-k B */
    NULL, //cholmod_dense  *Bdense,     /* dense  m-by-k B */
    /* outputs: */
    &Zsparse, //cholmod_sparse **Zsparse,   /* sparse Z */
    &Zdense, //cholmod_dense  **Zdense,    /* dense Z */
    &R, //cholmod_sparse **R,         /* e-by-n sparse matrix */
    &E, //int32_t **E,                /* size n column perm, NULL if identity */
    &H, //cholmod_sparse **H,         /* m-by-nh Householder vectors */
    &HPinv, //int32_t **HPinv,            /* size m row permutation */
    &HTau, //cholmod_dense **HTau,       /* 1-by-nh Householder coefficients */
    cc //cholmod_common *cc          /* workspace and parameters */
) ;

    if (!cholmod_print_sparse(R, "R matrix", cc)){
        return NULL;
    }

    if (!cholmod_print_sparse(H, "H matrix", cc)){
        return NULL;
    }

    if (!cholmod_print_dense(HTau, "HTau matrix", cc)){
        return NULL;
    }

    if (!cholmod_print_sparse(Zsparse, "Zsparse matrix", cc)){
        return NULL;
    }

    // if (!cholmod_print_dense(Zdense, "Zdense matrix", cc)){
    //     return NULL;
    // }


    /*To test with valgrind*/
    free(HPinv);
    free(E);
    cholmod_free_dense(&HTau, cc);
    cholmod_free_sparse(&H, cc);
    cholmod_free_sparse(&R, cc);
     cholmod_free_sparse(&input_matrix, cc);
     cholmod_free_sparse(&Zsparse, cc);



    if (!cholmod_finish(cc)){
        return NULL;
    }

    Py_RETURN_NONE;
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
    import_array(); /*Valgrind complains about this, but seems benign.*/
    return PyModuleDef_Init(&suitesparseqr);
}

