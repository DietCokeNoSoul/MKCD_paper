import os
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)
from typing import *

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from tqdm import trange

from random_walk.randomwalk import random_walk_sequence, adjmtx2adjlist


class DeepWalk(nn.Module):
    '''
        n: the number of nodes
        d: the embedding size
    '''
    def __init__(self, n, d) -> None:
        super().__init__()
        self.Phi = nn.Embedding(n, d)
        self.classifier = nn.Linear(d, n)

    def forward(self, i):
        x = self.Phi(i)
        x = self.classifier(x)
        return x


class SkipGram(Dataset):
    '''
    Skip-Gram

    path: path
    w: window size 

    https://arxiv.org/abs/1301.3781v3
    '''
    def __init__(self, path, w) -> None:
        super().__init__()
        n = len(path)
        X = []
        Y = []
        for i in range(n):
            st = max(0, i-w)
            ed = min(n, i+w)
            X.extend([path[i]]*(i-st))
            Y.extend(path[st:i])
            X.extend([path[i]]*(ed-i))
            Y.extend(path[i:ed])
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.nelement()


def deep_walk(G, w=10, d=128, γ=30, t=40, gpu: bool = False, batch_size=64) -> nn.Module:
    '''
    G: Graph G(N,E)
    w: window size 2
    d: embedding size 32
    γ: walks per vertex 30
    t: walk length 6

    https://doi.org/10.1145/2623330.2623732
    '''
    n, _ = G.shape
    device = torch.device("cuda") if gpu else torch.device("cpu")
    model = DeepWalk(n, d)
    O = np.arange(n)
    loss_fun = nn.CrossEntropyLoss()
    model.to(device)
    opt = Adam(model.parameters(), lr=0.003)
    for i in trange(γ):
        np.random.shuffle(O)
        for W in random_walk_sequence(G, t, 1, O):
            for a, b in DataLoader(SkipGram(W, w), batch_size=batch_size):
                predicts = model(a.to(device))
                loss = loss_fun(predicts, b.to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
    return model