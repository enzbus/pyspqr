from pyspqr import qr
import scipy as sp
import numpy as np

# def invert_permutation(p):
#     """Return an array s with which np.array_equal(arr[p][s], arr) is True.
#     The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
#     """
#     #p = np.asanyarray(p) # in case p is a tuple, etc.
#     s = np.empty_like(p)
#     s[p] = np.arange(len(p))
#     return s

def densify(linop):
    result = np.zeros(linop.shape)
    for j in range(result.shape[1]):
        e = np.zeros(result.shape[1])
        e[j] = 1.
        result[:,j] = linop @ e
    return result

def make_permutation_linop(permutation):

    n = len(permutation)
    #inverse = invert_permutation(permutation)

    def _matvec(vector):
        return vector[permutation]

    def _rmatvec(vector):
        result = np.empty_like(vector)
        result[permutation] = vector
        return result

    return sp.sparse.linalg.LinearOperator(
        shape=(n,n),
        matvec=_matvec,
        rmatvec=_rmatvec,
    )

    

def make_q(H, HTau, HPinv):

    def _q_rmatvec(input_vector):
        result = np.empty_like(input_vector)
        result[HPinv] = input_vector
        n_reflections = H.shape[1]
        for k in range(n_reflections):
            col = H[:,k].todense().A1
            result -= ((col @ result) * HTau[k]) * col
        return result

    def _q_matvec(input_vector):
        result = np.array(input_vector, copy=True)
        n_reflections = H.shape[1]
        for k in range(n_reflections)[::-1]:
            col = H[:,k].todense().A1
            result -= ((col @ result) * HTau[k]) * col
        return result[HPinv]

    return sp.sparse.linalg.LinearOperator(
        shape=(m,m),
        matvec=_q_matvec,
        rmatvec=_q_rmatvec,
    )

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    np.random.seed(0)
    m,n = 50,50
    while True:
        try:
            A = sp.sparse.random(m,n, density=.25, format='csc')
            R, H, HPinv, HTau, E = qr(A)
            assert not np.all(HPinv == np.arange(m, dtype=int))
            break
        except (ValueError, AssertionError):
            continue
    R = R.todense()
    Q = make_q(H, HTau, HPinv)
    npQ, npR = np.linalg.qr(A.todense())
    npQ[np.abs(npQ) < 1e-15] = 0.
    npR[np.abs(npR) < 1e-15] = 0.

    Elinop = densify(make_permutation_linop(E))

    # this is super important; means that indeed the rest of the result
    # is an orthogonal matrix
    assert np.allclose(np.linalg.svd(npR)[1], np.linalg.svd(R)[1])

    # here how the two permutations act
    assert np.allclose(densify(Q) @ R @ densify(make_permutation_linop(E)),  A.todense())

    assert np.allclose(densify(make_permutation_linop(E).T) @ R.T @ densify(Q.T),  A.todense().T)
    



