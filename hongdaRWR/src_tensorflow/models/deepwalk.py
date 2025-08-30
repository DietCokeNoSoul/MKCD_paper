from typing import *

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers

from ..random_walk.randomwalk import adjmtx2trans, random_walk_sequence
from .skipgrams import SkipGram, get_skipgram_pairs


def deepwalk_paths(s, a, t, r, l):
    '''
    s:  开始结点集合\n
    a:  邻接矩阵\n
    t:  转移矩阵\n
    r:  每个节点遍历次数\n
    l:  路径长度\n
    '''
    for _ in range(r):
        np.random.shuffle(s)
        for p in random_walk_sequence(a, s, t, l, 1):
            yield p


class DeepWalk(object):
    def __init__(self, graph, embedding_size) -> None:
        '''
        graph:        
        window_size:
        embedding_size:
        walk_length:
        walk_repeats:

        '''
        super().__init__()
        self.n_nodes = graph.shape[0]
        self.skipgram = SkipGram(self.n_nodes, embedding_size)
        self.nodes = np.arange(self.n_nodes)
        self.graph = graph
        self.transfer = adjmtx2trans(graph)

    def train(self, window_size,  walk_length, walk_repeats):
        paths = deepwalk_paths(self.nodes, self.graph, self.transfer, walk_repeats, walk_length)
        def generator():
            for each in get_skipgram_pairs(self.n_nodes, paths, window_size, True):
                yield each
        ds = tf.data.Dataset.from_generator(
            generator,
            args=[],
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        optimizer = optimizers.Adam()
        loss_func = losses.BinaryCrossentropy()
        for source, target in ds.batch(64):
            with tf.GradientTape() as tape:
                predicts = self.skipgram(source)
                loss = loss_func(target, predicts)
                gs = tape.gradient(loss, self.skipgram.trainable_variables)
            optimizer.apply_gradients(zip(gs, self.skipgram.trainable_variables))
            print(loss)
