from typing import *
import numpy as np
import os
import shutil

from ..metrics.metrics import Metrics


class CrossValidation(object):
    def __init__(self, k: int, n_samples: int) -> None:
        self.folds = k
        self.n_samples = n_samples

    def partition(self) -> List:
        self.samples = np.arange(self.n_samples)
        np.random.shuffle(self.samples)
        r = self.n_samples % self.folds
        p = self.n_samples // self.folds
        n = 0
        sp_list = []
        for i in range(self.folds):
            sp_list.append(n+p+1 if i < r else n+p)
            n += p+1 if i < r else p
        return np.split(self.samples, sp_list[:-1])


def _fold_dir(dir: str, idx: int) -> str:
    return os.path.join(dir, "fold_%d" % (idx+1))


def _cross_validation_check(k: int, RD: np.ndarray, dir: str) -> bool:
    if not os.path.exists(dir):
        return False
    path_rd = os.path.join(dir, "RD.npy")
    if not os.path.isfile(path_rd):
        return False
    rd = np.load(path_rd).astype(int)
    if not np.all(rd == RD.astype(int)):
        return False
    for i in range(k):
        path_fold = os.path.join(dir, "fold_%d" % (i+1))
        if not os.path.exists(path_fold):
            return False
        path_trains = os.path.join(path_fold, "train.mask.npy")
        if not os.path.exists(path_trains):
            return False
    return True


class DrugDiseaseCrossValidation(CrossValidation):
    def __init__(self, k: int, RD: np.ndarray, dir: str, neg_sampling: int) -> None:
        self.dir = dir
        self.neg_sampling = neg_sampling
        self.RD = RD.astype(int)
        positives = np.array(np.where(self.RD == 1)).T
        n_samples = positives.shape[0]
        if not os.path.exists(dir):
            os.mkdir(dir)
        super(DrugDiseaseCrossValidation, self).__init__(k, n_samples)
        if not _cross_validation_check(k, RD, dir):
            self.repartation()

    def repartation(self):
        '''重新划分测试集和训练集'''
        print("重新划分交叉数据集...")
        positives = np.array(np.where(self.RD == 1)).T
        negatives = np.array(np.where(self.RD == 0)).T
        n_samples = positives.shape[0]
        np.random.shuffle(positives)
        np.random.shuffle(negatives)
        shutil.rmtree(self.dir)
        os.mkdir(self.dir)
        np.save(os.path.join(self.dir, "RD.npy"), self.RD)
        for i, ids in enumerate(self.partition()):
            fold_dir = _fold_dir(self.dir, i)
            os.mkdir(fold_dir)
            mask = np.ones(n_samples, dtype=bool)
            mask[ids] = False
            negs = [negatives[i*n_samples:(i+1)*n_samples][mask, :]
                    for i in range(self.neg_sampling)]
            samples = np.concatenate(
                (positives[mask, :], *negs),
                axis=0
            )
            np.save(os.path.join(fold_dir, "train.mask.npy"), samples)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        path_fold = _fold_dir(self.dir, idx)
        trains = np.load(os.path.join(path_fold, "train.mask.npy"))
        rs = trains[:, 0]
        ds = trains[:, 1]
        rd = self.RD.copy()
        mask = np.ones_like(rd, dtype=bool)
        mask[rs, ds] = False
        tests = np.array(np.where(mask)).T
        rs = tests[:, 0]
        ds = tests[:, 1]
        rd[rs, ds] = False
        return rd, trains, tests

    def record_predictions(self, idx, scores):
        '''将预测评分保存'''
        path_fold = _fold_dir(self.dir, idx)
        np.save(os.path.join(path_fold, "scores.npy"), scores)

    def metrics(self, idx: Union[int, List[int], None] = None, combine_by_drug: bool = False):
        '''返回fpr, tpr, r, p向量'''
        if isinstance(idx, list) or idx is None:
            if idx is None:
                idx = range(self.folds)
            scores = np.zeros_like(self.RD, dtype=float)
            count = np.zeros_like(self.RD, dtype=int)
            for i in idx:
                path_fold = _fold_dir(self.dir, i)
                s = np.load(os.path.join(path_fold, "scores.npy"))
                _, _, tests = self.__getitem__(i)
                count[tests[:, 0], tests[:, 1]] += 1
                scores += s
            mask = count > 0
            count = np.where(count == 0, 1, count)
            scores = scores/count
        else:
            path_fold = _fold_dir(self.dir, idx)
            scores = np.load(os.path.join(path_fold, "scores.npy"))
            _, _, tests = self.__getitem__(idx)
            mask = np.zeros_like(self.RD, dtype=bool)
            mask[tests[:, 0], tests[:, 1]] = True

        if combine_by_drug:
            n_p = np.logical_and(self.RD == 1, mask)
            n_p = np.sum(n_p, axis=1, keepdims=True)
            useful = n_p > 0
            n_t = np.sum(mask, axis=1, keepdims=True)
            len_min = np.min(n_t)
            scores[np.logical_not(mask)] = 10
            ids = np.argsort(scores, axis=1)[:, ::-1]
            labels = np.take_along_axis(self.RD, ids, axis=1)[:, :len_min]
            scores = np.take_along_axis(scores, ids, axis=1)[:, :len_min]
            cms = np.zeros((len_min, 2, 2))
            for drug_line, label, flag in zip(scores, labels, useful[:, 0]):
                if not flag:
                    continue
                cm = Metrics.confusion_matrices(drug_line, label)
                cms += cm
        else:
            scores = scores[mask]
            labels = self.RD[mask]
            cms = Metrics.confusion_matrices(scores, labels)
        r = Metrics.recall(cms)
        p = Metrics.precision(cms)
        tpr = Metrics.TPR(cms)
        fpr = Metrics.FPR(cms)
        return fpr, tpr, r, p
