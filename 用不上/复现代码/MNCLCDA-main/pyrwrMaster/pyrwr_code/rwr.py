#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)
import numpy as np
from utils import iterator
from pyrwr_code import pyrwr


class RWR(pyrwr.PyRWR):
    def __init__(self):
        super().__init__()

    def compute(self,
                seed,
                c=0.15,
                epsilon=1e-6,
                max_iters=100,
                handles_deadend=True,
                device='gpu'):
        '''
        Compute the RWR score vector w.r.t. the seed node

        inputs
            seed : int
                seed (query) node id
            c : float
                restart probability
            epsilon : float
                error tolerance for power iteration
            max_iters : int
                maximum number of iterations for power iteration
            handles_deadend : bool
                if true, it will handle the deadend issue in power iteration
                otherwise, it won't, i.e., no guarantee for sum of RWR scores
                to be 1 in directed graphs
        outputs
            r : ndarray
                RWR score vector
        '''

        self.normalize()

        # adjust range of seed node id
        seed = seed - self.base

        #  q = np.zeros((self.n, 1))
        q = np.zeros(self.n)
        if seed < 0 or seed >= self.n:
            raise ValueError('Out of range of seed node id')

        q[seed] = 1.0

        r, residuals = iterator.iterate(self.nAT, q, c, epsilon,
                                        max_iters, handles_deadend,
                                        norm_type=1,
                                        device=device)

        return r
