import numpy as np
from typing import *
import matplotlib.pyplot as plt


def Fisher_Yates_shuffling(sim: np.ndarray) -> np.ndarray:
    s = sim.copy()
    np.random.shuffle(s)
    return s


def in_range(sim: np.ndarray, begin: float, end: float) -> np.ndarray:
    if end == 1:
        end = 1+1e-3
    return np.logical_and(sim >= begin, sim < end)


def count_range(sim: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ranges = np.linspace(0, 1, num=11)
    flags = [in_range(sim, ranges[i], ranges[i+1]) for i in range(10)]
    rates = []
    for flag in flags:
        rate = np.logical_and(flag, mask).sum()/flag.sum()
        rates.append(rate)
    return np.array(rates)


def analysis_of_similarity(sim: np.ndarray, mask: np.ndarray, n: int = 10) -> np.ndarray:
    randomized = []
    for i in range(n):
        s = Fisher_Yates_shuffling(sim)
        randomized.append(count_range(s, mask))
    hists = np.stack(randomized, axis=0)
    origin = count_range(sim, mask)
    return origin, hists.mean(axis=0)


def ClusterONE():
    pass


def similarity_hist(sim, mask, savefig: Optional[str] = "similarity.jpg"):
    ans = analysis_of_similarity(sim, mask, 10)
    labels = ["0~0.1", "0.1~0.2", "0.2~0.3", "0.3~0.4",
              "0.4~0.5", "0.5~0.6", "0.6~0.7", "0.7~0.8", "0.8~0.9", "0.9~1"]
    fig, ax = plt.subplots()
    ax.bar(range(10), ans[0]*100, 0.4, label="origin", color='tab:blue')
    ax.plot(range(10), ans[1]*100, label="randomized", color='tab:orange')
    ax.set_xlabel("Drug pairs similarity bins")
    ax.set_ylabel("Percentage of drug pairs sharing diseases")
    ax.legend()
    ax.set_xticks(range(10), labels, rotation=15)
    fig.tight_layout()
    if isinstance(savefig, str):
        plt.savefig(savefig)
    else:
        plt.show()


def normalization(M: np.ndarray) -> np.ndarray:
    D = M.sum(axis=1, keepdims=True)
    return M/(np.sqrt(D)@np.sqrt(D.T))


def MBiRW(R, D, A, alpha=0.3, l=2, r=2) -> np.ndarray:
    '''https://doi.org/10.1093/bioinformatics/btw228'''
    MR = normalization(R)
    MD = normalization(D)
    A = A/A.sum()
    RD = A
    for t in range(max(l, r)):
        rflag, dflag = 1, 1
        if t < l:
            Rr = alpha*MR@RD+(1-alpha)*A
            rflag = 1
        if t < r:
            Rd = alpha*RD@MD+(1-alpha)*A
            dflag = 1
        RD = (rflag*Rr+dflag*Rd)/(rflag+dflag)
    return RD
