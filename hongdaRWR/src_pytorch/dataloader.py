from typing import Iterable
from torch.utils.data import DataLoader, Dataset
from torch import from_numpy, cat, stack
from torch.nn import Module
from numpy import ndarray, array
from numpy import stack as np_stack
from torch.utils.data.sampler import Sampler

from utils import joint, rand_walk


class MultiDrugDisease(Dataset):
    '''药物疾病数据集

    Attributes:
        RD:真实的药物-疾病关联矩阵
        features:多种药物相似性、多种疾病相似性、假设出的已知药物疾病关联构成的
        len_R,len_D:药物和疾病的个数。
        pairs:数据集中药物-疾病对。
    '''

    def __init__(self, pairs: ndarray, RD: ndarray, RRs: list, DDs: list, labels: ndarray):
        '''初始化数据集'''
        super(MultiDrugDisease, self).__init__()
        self.labels = from_numpy(labels)
        self.RD = from_numpy(RD.copy())
        self.features = from_numpy(joint(RRs, DDs, RD))
        self.len_R, self.len_D = RD.shape
        self.pairs = from_numpy(array(pairs.copy()))

    def __getitem__(self, index):
        '''返回第index个药物疾病对特征矩阵'''
        r, d = self.pairs[index]
        r_d = self.get_feature(r, d)
        label = self.labels[r, d]
        return (r_d, label)

    def get_feature(self, r, d):
        '''返回药物r,疾病d的特征'''
        ri = self.features[:, r, :]
        dj = self.features[:, :, self.len_R+d]
        r_d = stack((ri, dj), dim=1)
        return r_d

    def __len__(self):
        '''数据集中所有样本个数'''
        return len(self.pairs)

    def to_device(self, device):
        self.RD = self.RD.to(device)
        self.features = self.features.to(device)

    def reset_labels(self, labels: ndarray):
        self.labels = from_numpy(labels)

    def reset_pairs(self, pairs: ndarray):
        self.pairs = from_numpy(pairs)


class RandomDrugDisease(Dataset):
    '''左路数据集（随机游走）

    Attributes:
        RD:真实的药物-疾病关联矩阵
        RRs:多种药物相似性
        DDs:多种疾病相似性
        alpha:重启概率
        steps:随机游走步长
        pairs:数据集中药物-疾病对。
    '''

    def __init__(self, pair_sampler, alpha: float, steps: int, RD: ndarray, RRs: list, DDs: list, labels: ndarray, mode="both"):
        '''初始化数据集'''
        super(RandomDrugDisease, self).__init__()
        assert mode in ["both", "left", "right"]
        self.mode = mode
        self.labels = from_numpy(labels)
        self.RD = from_numpy(RD.copy())
        self.len_R, self.len_D = RD.shape
        if isinstance(pair_sampler, Sampler):
            self.pairs = [pair_sampler.data_source[x] for x in pair_sampler]
        else:
            self.pairs = pair_sampler
        self.pair_sampler = pair_sampler
        self.RWs = from_numpy(
            np_stack(
                [rand_walk(alpha, steps, H) for H in joint(RRs, DDs, RD)]
            )
        ).transpose_(0,1)
        self.vis = [False for i in self.pairs]

    def features_sequences(self, r: int, d: int) -> ndarray:
        '''从随机游走矩阵序列中抽出药物d和疾病r的特征向量，构成特征张量'''
        ri = self.RWs[:, :, r, :]
        dj = self.RWs[:, :, d+self.len_R, :]
        return cat((ri, dj), dim=-1)

    def get_feature(self, r, d):
        '''返回药物r,疾病d的特征'''
        ri = self.RWs[0, :, r, :]
        dj = self.RWs[0, :, self.len_R+d, :]
        r_d = stack((ri, dj), dim=1)
        return r_d

    def __getitem__(self, index):
        '''获得第index个药物疾病对的特征张量'''
        self.vis[index] = True
        r, d = self.pairs[index]
        label = self.labels[r, d]
        if all(self.vis):
            if isinstance(self.pair_sampler, Sampler):
                self.pairs = [self.pair_sampler.data_source[x]
                              for x in self.pair_sampler]
            self.vis = [False for each in self.pairs]
        if self.mode == "both":
            return r, d, self.features_sequences(r, d), self.get_feature(r, d), label
        elif self.mode == "left":
            return r, d, self.features_sequences(r, d), label
        else:
            return r, d, self.get_feature(r, d), label

    def __len__(self):
        '''数据集中所有的样本数量'''
        return len(self.pairs)


class FusionDataset(Module):
    '''左右路融合数据集'''

    def __init__(self, left: RandomDrugDisease, right: MultiDrugDisease):
        assert (left.pairs == right.pairs).all(), "pairs must be same."
        super(FusionDataset, self).__init__()
        self.left = left
        self.right = right

    def __getitem__(self, index):
        r, d = self.left.pairs[index]
        feature_left = self.left.features_sequences(r, d)
        feature_right = self.right.get_feature(r, d)
        label = self.left.labels[r, d]
        return feature_left, feature_right, label

    def __len__(self):
        return len(self.left.pairs)

    def to_device(self, device):
        self.left.to_device(device)
        self.right.to_device(device)

    def reset_labels(self, labels: ndarray):
        self.left.reset_labels(labels)
        self.right.reset_labels(labels)

    def reset_pairs(self, pairs):
        self.left.reset_pairs(pairs)
        self.right.reset_pairs(pairs)
