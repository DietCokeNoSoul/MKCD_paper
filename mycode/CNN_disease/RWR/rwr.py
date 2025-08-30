import torch
import networkx as nx

class Data:
    def __init__(self, rg: torch.Tensor, sigama: float):
        self.rg = rg.float()
        self.sigama = sigama

def line_normalization(a: torch.Tensor) -> torch.Tensor:
    '''行归一化'''
    an = a.clone()
    s = a.sum(dim=1, keepdim=True)
    s[s == 0] = 1
    return an / s

def rand_walk(alpha: float, steps: int, A: torch.Tensor) -> list:
    '''以重启概率α自初始状态随机游走RD共step步，返回含初始状态的关系序列列表'''
    P = A.clone()
    P.fill_diagonal_(0)
    P = line_normalization(P)
    E = A
    ans = [E]
    W = E
    for s in range(steps):
        W = (1 - alpha) * torch.matmul(P.T, W) + alpha * E
        ans.append(W.T)
    return ans

# 将rand_walk函数的返回值数组，每一个数据将乘上一个参数，然后将所有数据相加，最后返回一个新的数组
def combine_rwr(weights: list, A: torch.Tensor) -> torch.Tensor:
    rwr = rand_walk(0.7, 2, A)
    sigama = 2e-3
    '''将rwr数组乘上weights数组，然后相加，返回一个新的数组'''
    rg = sum([w * weight for w, weight in zip(rwr, weights)])
    # 使用Data类，将rwr和sigama封装成一个Data对象
    return Data(rg=rg, sigama=sigama)

def bfs_topology_embedding(A: torch.Tensor, depth: int = 2) -> torch.Tensor:
    """
    对每个结点，使用BFS在邻接矩阵A上生成邻居拓扑嵌入。
    返回一个n*n的矩阵，每一行表示对应结点的邻居嵌入（1表示在BFS范围内，0表示不在）。
    """
    n = A.shape[0]
    embeddings = torch.zeros((n, n))
    G = nx.from_numpy_array(A.cpu().numpy(), create_using=nx.Graph)
    for node in range(n):
        neighbors = set([node])
        current_level = set([node])
        for _ in range(depth):
            next_level = set()
            for v in current_level:
                next_level.update(G.neighbors(v))
            neighbors.update(next_level)
            current_level = next_level
        for v in neighbors:
            embeddings[node, v] = 1
    return embeddings

def pagerank_topology_embedding(A: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
    """
    对每个结点，使用PageRank在邻接矩阵A上生成邻居拓扑嵌入。
    返回一个n*n的矩阵，每一行表示对应结点的PageRank分布（归一化概率）。
    """
    n = A.shape[0]
    G = nx.from_numpy_array(A.cpu().numpy(), create_using=nx.DiGraph)
    pr = nx.pagerank(G, alpha=alpha)
    embeddings = torch.zeros((n, n))
    for node in range(n):
        personalized = nx.pagerank(G, alpha=alpha, personalization={node: 1.0})
        for j in range(n):
            embeddings[node, j] = personalized.get(j, 0.0)
    return embeddings