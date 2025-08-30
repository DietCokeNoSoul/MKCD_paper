import numpy as np
from typing import *


class Metrics(object):

    @staticmethod
    def confusion_matrices(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        '''                
                           | actual positive | actual negative |\n
        predicted positive |        TP       |        FP       |\n
        predicted negative |        FN       |        TN       |\n

        '''
        n = scores.size
        labels = labels.astype(bool)
        matrices = np.empty((n, 2, 2))
        P = labels.sum()
        N = n-P
        TP, FP = P, N
        FN, TN = 0, 0
        ids = np.argsort(scores)
        for i in range(n):
            idx = ids[i]
            if labels[idx] == True:
                TP -= 1
                FN += 1
            else:
                TN += 1
                FP -= 1
            matrices[i, 0, 0] = TP
            matrices[i, 0, 1] = FP
            matrices[i, 1, 0] = FN
            matrices[i, 1, 1] = TN
        return matrices

    @staticmethod
    def TPR(CM: np.ndarray) -> np.ndarray:
        '''
        TP/(TP+FN)
        '''
        TP = CM[:, 0, 0]
        FN = CM[:, 1, 0]
        return TP/(TP+FN)

    @staticmethod
    def FPR(CM: np.ndarray) -> np.ndarray:
        '''
        FP/(FP+TN)
        '''
        FP = CM[:, 0, 1]
        TN = CM[:, 1, 1]
        return FP/(FP+TN)

    @staticmethod
    def recall(CM: np.ndarray) -> np.ndarray:
        '''
        TP/(TP+FN)
        '''
        return Metrics.TPR(CM)[:-1]

    @staticmethod
    def precision(CM: np.ndarray) -> np.ndarray:
        '''
        TP/(TP+FP)
        '''
        TP = CM[:-1, 0, 0]
        FP = CM[:-1, 0, 1]
        return TP/(TP+FP)

    @staticmethod
    def auc(x: np.ndarray, y: np.ndarray) -> float:
        check = np.all(x[:-1] <= x[1:]) or np.all(x[:-1] >= x[1:])
        assert check, "x is not sorted."
        a, b = 0, 0
        s = 0
        for i in np.argsort(x):
            s += (b+y[i])/2*(x[i]-a)
            a, b = x[i], y[i]
        return s
