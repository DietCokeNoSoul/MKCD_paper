import numpy as np


def mse(A: np.ndarray, B: np.ndarray) -> float:
    return np.sum((A-B) ^ 2)


def row_col_sum(A: np.ndarray) -> np.ndarray:
    n, m = A.shape
    row = A.sum(axis=1, keepdims=True)*np.ones(n, m)
    col = A.sum(axis=0, keepdims=True)*np.ones(n, m)
    return row+col


def SCMFDD(A: np.ndarray, R: np.ndarray, D: np.ndarray, k, u, λ) -> np.ndarray:
    '''
    SCMFDD
        A: Known drug-disease association matrix
        R: Drug similarity matrix
        D: Disease similarity matrix
        k: The dimension of drugs and diseases in low-rank spaces
        u, λ: The regularization parameter
    https://doi.org/10.1186/s12859-018-2220-4
    '''
    n, m = A.shape
    X = np.random.rand(n, k)
    Y = np.random.rand(m, k)
    I = np.eye(k, k)
    Wr = row_col_sum(R)
    Wd = row_col_sum(D)
    R = R+R.T
    D = D+D.T
    while True:
        X_ = np.zeros_like(X)
        Y_ = np.zeros_like(Y)
        P = X.T@X
        Q = Y.T@Y
        for i in range(n):
            p = A[i:i+1, :]@Y+λ*np.sum(X*Wr[:, i:i+1], axis=0)@I
            q = Q+u*I+λ*Wr[i, :].sum()@I
            X[i, :] = p@np.linalg.inv(q)
        for j in range(m):
            p = A[j:j+1, :]@X+λ*np.sum(Y*Wd[:, j:j+1], axis=0)@I
            q = P+u*I+λ*Wd[j, :].sum()@I
            Y[j, :] = p@np.linalg.inv(q)
        if mse(X, X_)+mse(Y, Y_) <= 1e-4:
            break
        else:
            X = X_
            Y = Y_
    Ap = X@Y.T
    return Ap
