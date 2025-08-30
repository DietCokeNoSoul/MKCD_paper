import os
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import numpy as np
from typing import *
from multiprocessing import Pool


def random_walk_(r: np.ndarray, M: np.ndarray) -> np.ndarray:
    return M @ r


def random_walk(state: np.ndarray, transition: np.ndarray, restart: np.ndarray, beta: float = 0.8, steps: int = 50, epsilon: float = 1e-6) -> np.ndarray:
    '''
    \nstate:       当前状态 `(N, 1)`或者`(N, N)`
    \ntransitioon: 转移概率矩阵`(N, N)`， 列向量为某节点跳向其它节点的概率
    \nrestart:     重启概率矩阵`(N, N)`, 列向量为某节点跳向其它节点的概率
    \nbeta:        沿当前节点走下去的概率，`1-beta`为从当前节点随机跳向任意其它节点的概率
    \nsteps:       随机游走最大步数
    \nepsilon:     收敛条件
    '''
    M = beta * transition + (1-beta) * restart
    r = state.copy()
    for step in range(steps):
        r_next = random_walk_(r, M)
        if np.linalg.norm(r-r_next) < epsilon:
            return r_next
        r = r_next
    return r


def random_walk_process(state: np.ndarray, transition: np.ndarray, restart: np.ndarray, beta: float = 0.8, steps: int = 50, epsilon: float = 1e-6) -> List[np.ndarray]:
    '''
    \nstate:       当前状态 `(N, 1)`或者`(N, N)`
    \ntransitioon: 转移概率矩阵`(N, N)`， 列向量为某节点跳向其它节点的概率
    \nrestart:     重启概率矩阵`(N, N)`, 列向量为某节点跳向其它节点的概率
    \nbeta:        沿当前节点走下去的概率，`1-beta`为从当前节点随机跳向任意其它节点的概率
    \nsteps:       随机游走最大步数
    \nepsilon:     收敛条件
    '''
    M = beta * transition + (1-beta) * restart
    r = state.copy()
    r_list = list()
    for step in range(steps):
        r_next = random_walk_(r, M)
        if np.linalg.norm(r-r_next) < epsilon:
            return r_next
        r = r_next
        r_list.append(r)
    return r_list


def adjmtx2adjlist(adj: np.ndarray, threshold: float = 1e-4) -> List[np.ndarray]:
    '''
    根据权重矩阵获得邻接表
    '''
    res = list()
    for line in adj:
        res.append(np.where(line > threshold)[0])
    return res


def adjmtx2trans(adj: np.ndarray) -> np.ndarray:
    '''
    根据邻接矩阵返回转移概率矩阵
    \n 返回的矩阵中，各列向量`v_i`表示`i`节点跳向其它节点的概率。
    \n `\sum_j v_{i,j} = 1`
    '''
    s = adj.sum(axis=0, keepdims=True)
    s = np.where(s == 0, 1, s)
    return adj/s


class RandomWalkerPool(object):
    '''
    多进程处理产生随机游走路径
    '''

    def __init__(self, adj: np.ndarray, trans: np.ndarray, length: int, walkers: int, repeats: int) -> None:
        '''
        \nadj:    邻接矩阵
        \ntrans:  转移概率矩阵
        \nlength： 随机游走路径长度
        \nwalkers:并行游走者个数
        \nrepeats:随机游走路径总数
        '''
        self.length = length
        self.adjlist = adjmtx2adjlist(adj)
        self.trans = trans if trans is not None else adjmtx2trans(adj)
        n_nodes = len(self.adjlist)
        srcs = np.random.choice(n_nodes, size=repeats)
        with Pool(walkers) as pool:
            self.output = pool.map(self.walker, srcs)

    def walker(self, start):
        path = [start]
        now = start
        while len(path) < self.length:
            neighbors = self.adjlist[now]
            p = self.trans[now][neighbors]
            neighbor = np.random.choice(neighbors, p=p/p.sum())
            path.append(neighbor)
            now = neighbor
        return path


def random_walk_sequence(adj: np.ndarray, length: int, repeats: int, srcs: Optional[np.ndarray] = None, threshold: float = 0.3, trans: Optional[np.ndarray] = None):
    '''
    产生随机游走路径
    \nadj:    邻接矩阵
    \ntrans:  转移概率矩阵
    \nlength： 随机游走路径长度
    \nrepeats: 每个源节点产生的随机游走路径个数
    '''
    adjlist = adjmtx2adjlist(adj, threshold)
    n_nodes = len(adjlist)
    trans = trans if trans is not None else adjmtx2trans(adj)

    def walker(start):
        path = [start]
        now = start
        while len(path) < length:
            neighbors = adjlist[now]
            p = trans[now][neighbors]
            neighbor = np.random.choice(neighbors, p=p/p.sum())
            path.append(neighbor)
            now = neighbor
        return path

    for _ in range(repeats):
        if srcs is None:
            srcs_ = np.random.choice(n_nodes, size=n_nodes)
        else:
            srcs_ = srcs
        for src in srcs_:
            yield walker(src)
