/******************************************************************************
* Copyright (C) 2024 Enzo Busseti
*
* This file is part of Pyspqr.
*
* Pyspqr is free software: you can redistribute it and/or modify it under the
* terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
*
* Pyspqr is distributed in the hope that it will be useful, but WITHOUT ANY
* WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
* A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* Pyspqr. If not, see <https://www.gnu.org/licenses/>.
*******************************************************************************

C extension exposing SuiteSparseQR factorization function.

Parsing Numpy arrays as inputs, expected already clean (contiguous, right
dtype, ...). Producing Numpy arrays as output.

Sadly it appears difficult to create arrays with memory allocated elsewhere (by
SuiteSparse), see for example:

https://stackoverflow.com/questions/28905659/creating-a-numpy-array-in-c-from-an-allocated-array-is-causing-memory-leaks

So we're creating fresh arrays and memcpy'ing the result of the SuiteSparse
objects, which we free here.

Excellent guide on CPython extension API:

https://pythonextensionpatterns.readthedocs.io/en/latest/

*/
#define PY_SSIZE_T_CLEAN
/*Can also be defined in setuptools.*/
#define Py_LIMITED_API 0x03060000
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
    if (!npy_arr){
        return NULL;
    }
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
    if (!data_arr){
        Py_DECREF(m_py);
        Py_DECREF(n_py);
        return NULL;
    }
    PyObject * indices_arr = create_1dim_array_from_data(
        matrix -> nzmax, NPY_INT32, sizeof(int32_t), matrix -> i);
    if (!indices_arr){
        Py_DECREF(m_py);
        Py_DECREF(n_py);
        Py_DECREF(data_arr);
        return NULL;
    }
    PyObject * indptr_arr = create_1dim_array_from_data(
        matrix -> ncol, NPY_INT32, sizeof(int32_t), matrix -> p);
    if (!indptr_arr){
        Py_DECREF(m_py);
        Py_DECREF(n_py);
        Py_DECREF(data_arr);
        Py_DECREF(indices_arr);
        return NULL;
    }
    PyObject *rslt = PyTuple_New(5);
    PyTuple_SetItem(rslt, 0, m_py);
    PyTuple_SetItem(rslt, 1, n_py);
    PyTuple_SetItem(rslt, 2, data_arr);
    PyTuple_SetItem(rslt, 3, indices_arr);
    PyTuple_SetItem(rslt, 4, indptr_arr);

    return rslt;
};

/*Numpy 1d array from 1-by-n CHOLMOD dense, doubles only.*/
static inline PyObject *
create_npyarr_from_cholmod_dense1d(
        cholmod_dense * matrix, /*Input matrix.*/
        cholmod_common * cc /*CHOLMOD workspace.*/
        ){
    if (!cholmod_check_dense(matrix, cc)){
        PyErr_SetString(PyExc_ValueError,
            "Tried to unpack malformed CHOLMOD dense matrix.");
        return NULL;
    }
    if (! (matrix->nrow == (size_t) 1)){
        PyErr_SetString(PyExc_ValueError,
            "Matrix nrow is not 1.");
        return NULL;
    }
    if (!matrix -> xtype == CHOLMOD_REAL){
        PyErr_SetString(PyExc_ValueError,
            "Only real CHOLMOD dense matrices are supported.");
        return NULL;
    }
    if (!matrix -> dtype == CHOLMOD_DOUBLE){
        PyErr_SetString(PyExc_ValueError,
            "Only double float CHOLMOD dense matrices are supported.");
        return NULL;
    }

    PyObject * data_arr = create_1dim_array_from_data(
        matrix -> ncol, NPY_DOUBLE, sizeof(double), matrix -> x);

    return data_arr;

};

static inline void * check_1d_double_arr(PyArrayObject * nparray){
    /*Check that object is valid 1-d double array.
    Set Python Exception and return NULL if not.*/

    if (!PyArray_Check(nparray) || PyArray_NDIM(nparray) != 1 || PyArray_TYPE(nparray) != NPY_DOUBLE || !PyArray_IS_C_CONTIGUOUS(nparray)){
        PyErr_SetString(PyExc_TypeError,
           "Data argument must be contiguous 1-dimensional double Numpy array.");
        return NULL;
    }
    return (void*)nparray;
}

static inline void * check_1d_int32_arr(PyArrayObject * nparray){
    /*Check that object is valid 1-d int32 array.
    Set Python Exception and return NULL if not.*/

    if (!PyArray_Check(nparray) || PyArray_NDIM(nparray) != 1 || PyArray_TYPE(nparray) != NPY_INT32 || !PyArray_IS_C_CONTIGUOUS(nparray)){
        PyErr_SetString(PyExc_TypeError,
           "Data argument must be contiguous 1-dimensional int32 Numpy array.");
        return NULL;
    }
    return (void*)nparray;
}

static inline PyObject * q_multiply(PyObject *self, PyObject *args){
    /*Compute forward or backward multiplication by Householder reflections.*/

    int m;
    int n_reflections;
    int backward;

    PyArrayObject *vector_np;
    PyArrayObject *householder_coefficients_np;
    PyArrayObject *householder_reflection_data_np;
    PyArrayObject *householder_reflection_indices_np;
    PyArrayObject *householder_reflection_indptr_np;

    double * restrict vector;
    double * restrict householder_coefficients;
    double * restrict householder_reflection_data;
    int * restrict householder_reflection_indices;
    int * restrict householder_reflection_indptr;

    PyArg_ParseTuple(args, "iiiOOOOO", &m, &n_reflections, &backward,
        &vector_np, &householder_coefficients_np,
        &householder_reflection_data_np,
        &householder_reflection_indices_np,
        &householder_reflection_indptr_np);

    if (PyErr_Occurred()) {
        return NULL;
    }

    if (check_1d_double_arr(vector_np) == NULL) {return NULL;}
    if (check_1d_double_arr(householder_coefficients_np) == NULL) {return NULL;}
    if (check_1d_double_arr(householder_reflection_data_np) == NULL) {return NULL;}
    if (check_1d_int32_arr(householder_reflection_indices_np) == NULL) {return NULL;}
    if (check_1d_int32_arr(householder_reflection_indptr_np) == NULL) {return NULL;}

    vector = PyArray_DATA(vector_np);
    householder_coefficients = PyArray_DATA(householder_coefficients_np);
    householder_reflection_data = PyArray_DATA(householder_reflection_data_np);
    householder_reflection_indices = PyArray_DATA(householder_reflection_indices_np);
    householder_reflection_indptr = PyArray_DATA(householder_reflection_indptr_np);

    double dot_product;
    if (!backward){
        for (int j = 0; j < n_reflections; j++){
            dot_product = 0.0;
            //printf("dot product value %f\n", dot_product);
            //printf("%d %d\n", householder_reflection_indptr[0], householder_reflection_indptr[1]);

            #pragma omp simd reduction(+:dot_product)
            for (int k = householder_reflection_indptr[j];
                k < householder_reflection_indptr[j+1]; k++){
                dot_product += householder_reflection_data[k] * vector[householder_reflection_indices[k]];
                //printf("dot product value %f\n", dot_product);
            }
            //printf("dot product value %f\n", dot_product);

            dot_product *= householder_coefficients[j];

            #pragma omp simd
            for (int k = householder_reflection_indptr[j];
                k < householder_reflection_indptr[j+1]; k++){
                vector[householder_reflection_indices[k]] -= householder_reflection_data[k] * dot_product;
                //printf("dot product value %f\n", dot_product);
            }
        }
    } else {
        for (int j = n_reflections-1; j >= 0; j--){
            dot_product = 0.0;
            //printf("dot product value %f\n", dot_product);
            //printf("%d %d\n", householder_reflection_indptr[0], householder_reflection_indptr[1]);

            #pragma omp simd reduction(+:dot_product)
            for (int k = householder_reflection_indptr[j];
                k < householder_reflection_indptr[j+1]; k++){
                dot_product += householder_reflection_data[k] * vector[householder_reflection_indices[k]];
                //printf("dot product value %f\n", dot_product);
            }
            //printf("dot product value %f\n", dot_product);

            dot_product *= householder_coefficients[j];

            #pragma omp simd
            for (int k = householder_reflection_indptr[j];
                k < householder_reflection_indptr[j+1]; k++){
                vector[householder_reflection_indices[k]] -= householder_reflection_data[k] * dot_product;
                //printf("dot product value %f\n", dot_product);
            }
        }
    }

    Py_RETURN_NONE;
};

static inline PyObject *qr(PyObject *self, PyObject *args){
    /* Wrap of QR factorization function in SuiteSparseQR.
    
    Only doubles CSC matrices with int32 indexing, which is what Scipy uses.

    Here SuiteSparseQR does memory allocation (only function in this module).
    Peak memory usage here might be 2/3x the size of the output memory size.

    Memory size of the output can't be determined easily, it depends of the
    sparse pattern. Experimentally, with structured matrices coming from convex
    programs, it should be OK, with the R and Q matrices about the same total
    size of the input matrix. It could be much worse if the input matrix has
    random sparse pattern.

    This requires a version of SuiteSparseQR around what was published in 2024
    (not sure which exact one), but around 2023 some changes were pushed in the
    C header file. In compilation to be safe we compile the latest mid-2024
    version.

    We return *only* objects allocated by CPython and free everything else. 

    Input is (m, n, matrix.data, matrix.indices, matrix.indptr)

    where matrix is a scipy.sparse.csc_matrix.
    */

    int m;
    int n;
    int ordering;
    PyArrayObject *data_np;
    PyArrayObject *indices_np;
    PyArrayObject *indptr_np;

    /*Parse and validate inputs.*/

    PyArg_ParseTuple(args, "iiOOOi", &m, &n, &data_np, &indices_np, &indptr_np, &ordering);
    
    if (PyErr_Occurred()) {
        return NULL;
    }

    if (check_1d_double_arr(data_np) == NULL) {return NULL;}
    if (check_1d_int32_arr(indices_np) == NULL) {return NULL;}
    if (check_1d_int32_arr(indptr_np) == NULL) {return NULL;}

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

    /* Initialize SuiteSparse QR.*/

    cholmod_common Common, *cc;
    cc = &Common;
    if (!cholmod_start(cc)){
        PyErr_SetString(PyExc_ValueError,
            "SuiteSparseQR couldn't be initialized!");
        return NULL;
    }

    /* Next section could instead use this code, which allocates the matrix
    using SuiteSparse:

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

    memcpy(input_matrix->x, PyArray_DATA(data_np), nnz*sizeof(double));
    memcpy(input_matrix->i, PyArray_DATA(indices_np), nnz*sizeof(int32_t));
    memcpy(input_matrix->p, PyArray_DATA(indptr_np), (n+1)*sizeof(int32_t));

    Remember to free it afterwards!

    */

    /*Create input matrix in SuiteSparse.*/

    cholmod_sparse input_matrix_struct = {
        .nrow = (size_t) m,
        .ncol = (size_t) n,
        .nzmax = (size_t) nnz,
        .p = PyArray_DATA(indptr_np),
        .i = PyArray_DATA(indices_np),
        .nz = NULL,
        .x = PyArray_DATA(data_np),
        .z = NULL,
        .stype = 0,
        .itype = CHOLMOD_INT,
        .xtype = CHOLMOD_REAL,
        .dtype = CHOLMOD_DOUBLE,
        .sorted = true,
        .packed = true,
    };
    cholmod_sparse *input_matrix = &input_matrix_struct;

    /*Validate input matrix.*/

    if (!cholmod_check_sparse(input_matrix, cc)){
        PyErr_SetString(PyExc_ValueError,
            "Input matrix failed validation!");
        cholmod_finish(cc);
        return NULL;
    }

    cholmod_print_sparse(input_matrix, "input matrix", cc);

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
    ordering, //int ordering,               /* all, except 3:given treated as 0:fixed */
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

    if (rank < 0){
        PyErr_SetString(PyExc_MemoryError,
            "SuiteSparseQR factorization returned error code! Probably there's not enough memory.");
        goto free_and_exit_with_exception;
    }

    printf("Rank of input matrix is %d\n", rank);

    /*Safety checks. These should never be triggered, if the input matrix is
    valid and there's enough memory SuiteSparse always returns a solution. In
    case they were triggered (i.e., some bug in SuiteSparse), the state of the
    output objects would be undefined. We still free everything.*/
    
    if (!cholmod_check_sparse(R, cc)){
        PyErr_SetString(PyExc_ValueError,
            "Result matrix R failed validation!");
        goto free_and_exit_with_exception;
    }

    cholmod_print_sparse(R, "R matrix", cc);

    if (!cholmod_check_sparse(H, cc)){
        PyErr_SetString(PyExc_ValueError,
            "Result matrix H failed validation!");
        goto free_and_exit_with_exception;
    }

    cholmod_print_sparse(H, "H matrix", cc);

    if (!cholmod_check_dense(HTau, cc)){
        PyErr_SetString(PyExc_ValueError,
            "Result matrix HTau failed validation!");
        goto free_and_exit_with_exception;
    }

    cholmod_print_dense(HTau, "HTau matrix", cc);

    /*We start freeing incrementally, to save memory. These are unused.*/
    cholmod_free_sparse(&Zsparse, cc);
    cholmod_free_dense(&Zdense, cc);

    /*Box Python objects to return.*/
    /*Depending on the ordering, E can be NULL.*/
    PyObject* E_np_or_none;
    if (!E){
        E_np_or_none = Py_None;
    } else {
    E_np_or_none = create_1dim_array_from_data(
        (size_t)n, NPY_INT32, sizeof(int32_t), (void*)E);
    free(E);
    }

    if (!E_np_or_none){
        /*Exception set by Numpy.*/
        free(HPinv);
        cholmod_free_dense(&HTau, cc);
        cholmod_free_sparse(&R, cc);
        cholmod_free_sparse(&H, cc);
        cholmod_finish(cc);
        return NULL;
    }

    PyObject* HPinv_np = create_1dim_array_from_data(
        (size_t)m, NPY_INT32, sizeof(int32_t), (void*)HPinv);
    free(HPinv);
    if (!HPinv_np){
        /*Exception set by Numpy.*/
        cholmod_free_dense(&HTau, cc);
        cholmod_free_sparse(&R, cc);
        cholmod_free_sparse(&H, cc);
        cholmod_finish(cc);
        return NULL;
    }

    PyObject * HPTau_np = create_npyarr_from_cholmod_dense1d(HTau, cc);
    cholmod_free_dense(&HTau, cc);
    if (!HPTau_np){
        /*Exception set by Numpy.*/
        Py_DECREF(HPinv_np);
        cholmod_free_sparse(&R, cc);
        cholmod_free_sparse(&H, cc);
        cholmod_finish(cc);
        return NULL;
    }

    PyObject * H_py = tuple_from_cholmod_sparse(H, cc);
    cholmod_free_sparse(&H, cc);
    if (!H_py){
        /*Exception set by Numpy.*/
        Py_DECREF(HPinv_np);
        Py_DECREF(HPTau_np);
        cholmod_free_sparse(&R, cc);
        cholmod_finish(cc);
        return NULL;
    }

    PyObject * R_py = tuple_from_cholmod_sparse(R, cc);
    cholmod_free_sparse(&R, cc);
    if (!R_py){
        /*Exception set by Numpy.*/
        Py_DECREF(HPinv_np);
        Py_DECREF(HPTau_np);
        Py_DECREF(H_py);
        cholmod_finish(cc);
        return NULL;
    }

    cholmod_finish(cc);
    PyObject *rslt = PyTuple_New(5);
    PyTuple_SetItem(rslt, 0, R_py);
    PyTuple_SetItem(rslt, 1, H_py);
    PyTuple_SetItem(rslt, 2, HPinv_np);
    PyTuple_SetItem(rslt, 3, HPTau_np);
    PyTuple_SetItem(rslt, 4, E_np_or_none);
    return rslt;

    /*Exception during QR factorization.*/
    free_and_exit_with_exception:

        cholmod_free_sparse(&Zsparse, cc);
        cholmod_free_dense(&Zdense, cc);
        free(HPinv);
        free(E);
        cholmod_free_dense(&HTau, cc);
        cholmod_free_sparse(&R, cc);
        cholmod_free_sparse(&H, cc);
        cholmod_finish(cc);
        return NULL;

};

/* Create the CPython module object.*/

static PyMethodDef methods[] = {
    {
        "qr",
        (PyCFunction) qr,
        METH_VARARGS,
        "Perform sparse QR decomposition."
    },
    {
        "q_multiply",
        (PyCFunction) q_multiply,
        METH_VARARGS,
        "Multiply forward or backward by sequence of Householder reflections."
    },
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyModuleDef _pyspqr = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_pyspqr",
    .m_doc = "Python bindings for SuiteSparseQR, internal module.",
    .m_size = 0,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit__pyspqr(void)
{   
    /*Valgrind complains about this, but seems benign.*/
    import_array();
    return PyModule_Create(&_pyspqr);
}

