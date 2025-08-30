import numpy as np
from sklearn.utils.validation import (check_array, check_symmetric, check_consistent_length)


def _flatten(messy):
    """
    Flattens a messy list of mixed iterables / not-iterables

    Parameters
    ----------
    messy : list of ???
        Combined list of iterables / non-iterables

    Yields
    ------
    data : ???
        Entries from `messy`

    Notes
    -----
    Thanks to https://stackoverflow.com/a/2158532 :chef-kissing-fingers-emoji:
    """
    for m in messy:
        if isinstance(m, (list, tuple)):
            yield from _flatten(m)
        else:
            yield m

def _check_SNF_inputs(aff):
    """
    Confirms inputs to SNF are appropriate

    Parameters
    ----------
    aff : `m`-list of (N x N) array_like
        Input similarity arrays. All arrays should be square and of equal size.
    """
    prep = []
    for a in _flatten(aff):
        ac = check_array(a, force_all_finite=True, copy=True) # force_all_finite=True是指检查是否有无穷大或NaN值，如果有则抛出ValueError异常，copy=True是指复制数据，防止修改原数据；函数返回的是
        prep.append(check_symmetric(ac, raise_warning=False))
    check_consistent_length(*prep)

    # TODO: actually do this check for missing data
    nanaff = len(prep) - np.sum([np.isnan(a) for a in prep], axis=0)
    if np.any(nanaff == 0):
        pass

    return prep

def _find_dominate_set(W, K=20):
    """
    Retains `K` strongest edges for each sample in `W`

    Parameters
    ----------
    W : (N, N) array_like
        Input data
    K : (0, N) int, optional
        Number of neighbors to retain. Default: 20

    Returns
    -------
    Wk : (N, N) np.ndarray
        Thresholded version of `W`
    """
    # let's not modify W in place
    Wk = W.copy()

    # determine percentile cutoff that will keep only `K` edges for each sample
    # remove everything below this cutoff
    cutoff = 100 - (100 * (K / len(W)))
    Wk[Wk < np.percentile(Wk, cutoff, axis=1, keepdims=True)] = 0

    # normalize by strength of remaining edges
    Wk = Wk / np.nansum(Wk, axis=1, keepdims=True)

    return Wk


def _B0_normalized(W, alpha=1.0):
    """
    Adds `alpha` to the diagonal of `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF
    alpha : (0, 1) float, optional
        Factor to add to diagonal of `W` to increase subject self-affinity.
        Default: 1.0

    Returns
    -------
    W : (N, N) np.ndarray
        Normalized similiarity array
    """
    # add `alpha` to the diagonal and symmetrize `W`
    W = W + (alpha * np.eye(len(W)))
    W = check_symmetric(W, raise_warning=False)

    return W


def snf(*aff, K=20, t=20, alpha=1.0):
    r"""
    Performs Similarity Network Fusion on `aff` matrices

    Parameters
    ----------
    *aff : (N, N) array_like
        Input similarity arrays; all arrays should be square and of equal size. 中文意思：输入相似性矩阵，所有数组都应该是方阵且大小相等。
    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. Default: 20
    t : int, optional
        Number of iterations to perform information swapping. Default: 20
    alpha : (0, 1) float, optional
        Hyperparameter normalization factor for scaling. Default: 1.0

    Returns
    -------
    W: (N, N) np.ndarray
        Fused similarity network of input arrays

    Notes
    -----
    In order to fuse the supplied :math:`m` arrays, each must be normalized. A
    traditional normalization on an affinity matrix would suffer from numerical
    instabilities due to the self-similarity along the diagonal; thus, a
    modified normalization is used:

    .. math::

       \mathbf{P}(i,j) =
         \left\{\begin{array}{rr}
           \frac{\mathbf{W}_(i,j)}
                 {2 \sum_{k\neq i}^{} \mathbf{W}_(i,k)} ,& j \neq i \\
                                                       1/2 ,& j = i
         \end{array}\right.

    Under the assumption that local similarities are more important than
    distant ones, a more sparse weight matrix is calculated based on a KNN
    framework:

    .. math::

       \mathbf{S}(i,j) =
         \left\{\begin{array}{rr}
           \frac{\mathbf{W}_(i,j)}
                 {\sum_{k\in N_{i}}^{}\mathbf{W}_(i,k)} ,& j \in N_{i} \\
                                                         0 ,& \text{otherwise}
         \end{array}\right.

    The two weight matrices :math:`\mathbf{P}` and :math:`\mathbf{S}` thus
    provide information about a given patient's similarity to all other
    patients and the `K` most similar patients, respectively.

    These :math:`m` matrices are then iteratively fused. At each iteration, the
    matrices are made more similar to each other via:

    .. math::

       \mathbf{P}^{(v)} = \mathbf{S}^{(v)}
                          \times
                          \frac{\sum_{k\neq v}^{}\mathbf{P}^{(k)}}{m-1}
                          \times
                          (\mathbf{S}^{(v)})^{T},
                          v = 1, 2, ..., m

    After each iteration, the resultant matrices are normalized via the
    equation above. Fusion stops after `t` iterations, or when the matrices
    :math:`\mathbf{P}^{(v)}, v = 1, 2, ..., m` converge.

    The output fused matrix is full rank and can be subjected to clustering and
    classification.
    """

    aff = _check_SNF_inputs(aff)
    Wk = [0] * len(aff)
    Wsum = np.zeros(aff[0].shape)

    # get number of modalities informing each subject x subject affinity
    n_aff = len(aff) - np.sum([np.isnan(a) for a in aff], axis=0)

    for n, mat in enumerate(aff):
        # normalize affinity matrix based on strength of edges
        mat = mat / np.nansum(mat, axis=1, keepdims=True)
        aff[n] = check_symmetric(mat, raise_warning=False)
        # apply KNN threshold to normalized affinity matrix
        Wk[n] = _find_dominate_set(aff[n], int(K))

    # take sum of all normalized (not thresholded) affinity matrices
    Wsum = np.nansum(aff, axis=0)

    for iteration in range(t):
        for n, mat in enumerate(aff):
            # temporarily convert nans to 0 to avoid propagation errors
            nzW = np.nan_to_num(Wk[n])
            aw = np.nan_to_num(mat)
            # propagate `Wsum` through masked affinity matrix (`nzW`)
            aff0 = nzW @ (Wsum - aw) @ nzW.T / (n_aff - 1)  # TODO: / by 0
            # ensure diagonal retains highest similarity
            aff[n] = _B0_normalized(aff0, alpha=alpha)

        # compute updated sum of normalized affinity matrices
        Wsum = np.nansum(aff, axis=0)

    # all entries of `aff` should be identical after the fusion procedure
    # dividing by len(aff) is hypothetically equivalent to selecting one
    # however, if fusion didn't converge then this is just average of `aff`
    W = Wsum / len(aff)

    # normalize fused matrix and update diagonal similarity
    W = W / np.nansum(W, axis=1, keepdims=True)  # TODO: / by NaN
    W = (W + W.T + np.eye(len(W))) / 2 # eye是单位矩阵

    return W