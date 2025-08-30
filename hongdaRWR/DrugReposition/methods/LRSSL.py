import numpy as np
import numpy.linalg as linalg

'''https://doi.org/10.1093/bioinformatics/btw770'''


def graph_model(X: np.ndarray, topk=10) -> np.ndarray:
    '''
    Eq.(1) and Eq.(2)
    note:
        `S` must be a similarity matrix
    '''
    ids = np.argpartition(X, -topk, axis=1)
    S = np.zeros_like(X)
    np.put_along_axis(S, ids[:, topk+1], 1, axis=1)
    return S


def laplacian(X: np.ndarray) -> np.ndarray:
    '''Eq.(3)'''
    d = X.sum(axis=1)
    D = np.diag(d)
    return D-X


def cosine(X: np.ndarray) -> np.ndarray:
    '''X: d*n'''
    T = X.T@X
    m = np.sqrt(T.diagonal()).reshape(1, T.shape[0])
    S = T/(m.T@m)
    return S


def LRSSL(Xs, Y: np.ndarray, λ=1.5, u=0.1, y=3, k=10, ites=2):
    '''
    https://doi.org/10.1093/bioinformatics/btw770

    Xs:     drug feature matrix, shape (*,n).
    Y:      drug-disease matrix, shape (n,c).
    λ:      regularization parameter for sparsity on Gp.
    u:      imbance parameter for drug-disease predict in entire objective function.
    y:      The parameter y > 1 is used to ensure that all the graph models contribute to the drug indication prediction and facilitate comprehensive understanding of the
results.
    k:      k neighbor.

    '''
    # ------------------------- Init begin --------------------------
    c, n = Y.shape
    m = len(Xs)
    Lps = np.array([
        laplacian(
            graph_model(cosine(Xp), k)
        ) for Xp in Xs
    ])
    Ly = laplacian(graph_model(cosine(Y.T), k))
    Gs = [np.random.rand(Xp.shape[1], c) for Xp in Xs]
    I = np.eye(n)
    a = np.ones(m+1)/(m+1)
    F = np.zeros_like(Y)
    # ------------------------- Init end ----------------------------

    for _ in range(ites):
        # -------------------- L define begin -----------------------
        L = a[m]*Ly
        for i in range(m):
            L = L+a[i]*Lps[i]
        # -------------------- L define end -------------------------

        P = linalg.inv(L+(1+m*u)*I)  # Eq. (10)

        def A(Xp: np.ndarray) -> np.ndarray:
            '''Eq. (18)'''
            return Xp@(u*I+u*u*P.T)@Xp.T+λ

        def B(Xp: np.ndarray) -> np.ndarray:
            '''Eq. (19)'''
            Bp = u*Xp@P@Y
            for q, Xq in enumerate(Xs):
                if np.all(Xq == Xp):
                    continue
                Bp = Bp+u*u*Xp@P.T@Xq.T@Gs[q]
            return Bp
        # -------------------- Eq. (17) begin -----------------------
        Q = Y
        for i in range(m):
            Ap = A(Xs[i])
            Bp = B(Xs[i])
            Gs[i] = Gs[i]*np.sqrt(
                np.divide(
                    np.negative(Ap@Gs[i])+np.positive(Bp),
                    np.positive(Ap@Gs[i])+np.negative(Bp)
                )
            )
            Q = Q+u*Xs[i].T@Gs[i]
        # -------------------- Eq. (17) end -------------------------

        F = P@Q     # Eq. (12)

        # -------------------- Eq. (22) begin -----------------------
        for i in range(m):
            a[i] = np.trace(F.T@Lps[i]@F)
        a[m] = np.trace(F.T@Ly@F)
        a = a ^ (1/(y-1))
        a = a/a.sum()
        # -------------------- Eq. (22) end -------------------------
    return F
