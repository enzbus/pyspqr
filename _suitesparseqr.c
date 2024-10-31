#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <SuiteSparseQR_C.h>
#include <stdbool.h>

/*Create 1-dim Numpy array from buffer memcopying data.*/
static inline PyObject *
create_1dim_array_from_data(
        size_t size, /*Numpy len of the array.*/
        int npy_dtype, /*NPY_DOUBLE, NPY_INT64, ....*/ 
        size_t bytesize, /*sizeof(double), ....*/ 
        void *data /*Pointer to data.*/){
    size_t dims[1];
    dims[0] = size;
    PyObject * npy_arr = PyArray_SimpleNew(1, dims, npy_dtype);
    memcpy(PyArray_DATA((PyArrayObject *) npy_arr), data,  size*bytesize);
    return npy_arr;
};

/*Unpack CHOLMOD sparse, return tuple (m,n,data,indices,indptr) with int32.

CAREFUL! CHOLMOD doesn't store the last value of indptr, which is nnz, that
instead Scipy stores. We're not adapting to Scipy format here. 
*/
static inline PyObject *
tuple_from_cholmod_sparse(
        cholmod_sparse * matrix, /*Input matrix.*/
        cholmod_common * cc /*CHOLMOD workspace.*/
        )
{
    size_t m,n,nnz;
    if (!cholmod_check_sparse(matrix, cc)){
        PyErr_SetString(PyExc_ValueError,
            "Tried to unpack malformed CHOLMOD sparse matrix.");
        return NULL;
    }
    if (!matrix -> itype == CHOLMOD_INT){
        PyErr_SetString(PyExc_ValueError,
            "Only int32 CHOLMOD sparse matrices are supported.");
        return NULL;
    }
    if (!matrix -> xtype == CHOLMOD_REAL){
        PyErr_SetString(PyExc_ValueError,
            "Only real CHOLMOD sparse matrices are supported.");
        return NULL;
    }
    if (!matrix -> dtype == CHOLMOD_DOUBLE){
        PyErr_SetString(PyExc_ValueError,
            "Only double float CHOLMOD sparse matrices are supported.");
        return NULL;
    }
    PyObject * m_py = PyLong_FromSsize_t(matrix -> nrow);
    PyObject * n_py = PyLong_FromSsize_t(matrix -> ncol);
    PyObject * data_arr = create_1dim_array_from_data(
        matrix -> nzmax, NPY_DOUBLE, sizeof(double), matrix -> x);
    PyObject * indices_arr = create_1dim_array_from_data(
        matrix -> nzmax, NPY_INT32, sizeof(int32_t), matrix -> i);
    PyObject * indptr_arr = create_1dim_array_from_data(
        matrix -> ncol, NPY_INT32, sizeof(int32_t), matrix -> p);

    PyObject *rslt = PyTuple_New(5);
    PyTuple_SetItem(rslt, 0, m_py);
    PyTuple_SetItem(rslt, 1, n_py);
    PyTuple_SetItem(rslt, 2, data_arr);
    PyTuple_SetItem(rslt, 3, indices_arr);
    PyTuple_SetItem(rslt, 4, indptr_arr);

    return rslt;
};

/*Numpy 1d array from 1d CHOLMOD dense, doubles only.*/
// static inline PyObject *
// create_npyarr_from_cholmod_dense1d(
//         cholmod_sparse * matrix, /*Input matrix.*/
//         cholmod_common * cc /*CHOLMOD workspace.*/
//         )

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
        PyErr_SetString(PyExc_ValueError,
            "Input matrix couldn't be created!");
        return NULL;
    }

    memcpy(input_matrix->x, data_arr, nnz*sizeof(double));
    memcpy(input_matrix->i, indices_arr, nnz*sizeof(int32_t));
    memcpy(input_matrix->p, indptr_arr, (n+1)*sizeof(int32_t));



    if (!cholmod_check_sparse(input_matrix, cc)){
        PyErr_SetString(PyExc_ValueError,
            "Input matrix failed validation!");
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
    SPQR_ORDERING_AMD, //int ordering,               /* all, except 3:given treated as 0:fixed */
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

    printf("Rank of input matrix is %d\n", rank);

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
    cholmod_free_sparse(&input_matrix, cc);
    cholmod_free_sparse(&Zsparse, cc);


    /*Box Python objects to return.*/
    PyObject* HPinv_np = create_1dim_array_from_data(
        (size_t)m, NPY_INT32, sizeof(int32_t), (void*)HPinv);
    free(HPinv);

    if (!E){
        PyErr_SetString(PyExc_ValueError,
            "E is not null!");
        return NULL;    }

    free(E);

    // size_t dims1[1];
    // dims1[0] = n;
    // PyObject * E_np = PyArray_SimpleNew(1, dims1, NPY_INT32);
    // memcpy(PyArray_DATA((PyArrayObject *) E_np), E,  (n)*sizeof(int32_t));
    // free(E);

    cholmod_free_dense(&HTau, cc);

    PyObject * H_py = tuple_from_cholmod_sparse(H, cc);
    cholmod_free_sparse(&H, cc);

    PyObject * R_py = tuple_from_cholmod_sparse(R, cc);
    cholmod_free_sparse(&R, cc);

    if (!cholmod_finish(cc)){
        return NULL;
    }

    PyObject *rslt = PyTuple_New(3);
    PyTuple_SetItem(rslt, 0, HPinv_np);
    PyTuple_SetItem(rslt, 1, R_py);
    PyTuple_SetItem(rslt, 2, H_py);
    return rslt;
};



static PyMethodDef methods[] = {
    {"qr", qr, METH_VARARGS, "Perform sparse QR decomposition."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _suitesparseqr = {
    PyModuleDef_HEAD_INIT,
    "_suitesparseqr",
    "Python bindings for SuiteSparseQR, internal module.",
    0,
    methods,
};

PyMODINIT_FUNC PyInit__suitesparseqr(void)
{   
    /*Valgrind complains about this, but seems benign.*/
    import_array();
    return PyModuleDef_Init(&_suitesparseqr);
}

