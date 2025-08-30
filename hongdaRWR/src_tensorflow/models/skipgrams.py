from typing import *
from math import ceil, log2

from tensorflow.keras import layers, Model, preprocessing

from ..random_walk.randomwalk import random_walk_sequence
from ..utils.mathematics import bitlist


def hierarchical_hight(max_n: int) -> int:
    '''采用树状编码后，树的深度
    '''
    return ceil(log2(max_n))


class SkipGramPairWapper(object):
    '''
    SkipGram pair的处理函数
    '''
    def __call__(self, source, target) -> tuple:
        raise NotImplementedError()


class NotProcess(SkipGramPairWapper):
    '''
        不处理
    '''
    def __call__(self, source, target) -> tuple:
        return source, target


class Hierarchical(SkipGramPairWapper):
    '''
    源结点仍为节点id，目的结点的形式为树状编码
    '''
    def __init__(self,fixlen) -> None:
        self.fixlen=fixlen
    def __call__(self, source, target) -> tuple:
        return source, bitlist(target,fix_length=self.fixlen)


def get_skipgram_pairs(n_nodes, paths, window_size, hierarchical: bool = True):
    '''
    根据游走路径创建skip gram对，作为模型的输入
    n_nodes：   结点个数\n
    paths:     路径集合\n
    window_size:SkipGram窗口\n
    hierarchical：是否使用层次化的树状编码处理目的结点的id\n
    '''
    h = hierarchical_hight(n_nodes)
    for path in paths:
        pairs, _ = preprocessing.sequence.skipgrams(
            path, vocabulary_size=n_nodes, window_size=window_size,
            negative_samples=0,
        )
        wapper = Hierarchical(fixlen=h) if hierarchical else NotProcess()
        for src, dst in pairs:
            yield wapper(src, dst)


class SkipGram(Model):

    def __init__(self, n_nodes: int, n_dims: int, *args, **kwargs):
        '''
        n_nodes:    结点总数
        n_dims:     嵌入编码特征向量维度
        '''
        super(SkipGram, self).__init__(*args, **kwargs)
        self.embedding = layers.Embedding(
            input_dim=n_nodes,
            output_dim=n_dims,
            embeddings_initializer='uniform'
        )
        n_class = hierarchical_hight(n_nodes)
        self.hierarchical_embedding = layers.Dense(
            n_class,
            # activation='sigmoid',
            activation=None,
        )

    def call(self, inputs, training=None, mask=None):
        w = self.embedding(inputs)
        tree = self.hierarchical_embedding(w)
        return tree
