import torch

def compute_gip_kernel(association_matrix):
    """
    计算 GIP Kernel 相似性矩阵

    参数
    ----------
    association_matrix : torch.Tensor
        circRNA 和 drug 的关联矩阵

    返回值
    -------
    gip_kernel : torch.Tensor
        circRNA 的 GIP Kernel 相似性矩阵
    """
    # 计算 gamma
    gamma = 1 / torch.mean(torch.sum(association_matrix ** 2, dim=1)) # 计算 gamma, 1 / (1 / n * sum(||x||^2)), x 为行向量, n 为行数, ||x||^2 为 x 的平方和, sum(||x||^2) 为所有行的平方和

    # 计算 GIP Kernel 相似性矩阵
    sq_dists = torch.cdist(association_matrix, association_matrix, p=2) ** 2 # 每个元素是两个circRNA之间的距禮的平方, p=2表示欧式距离 
    gip_kernel = torch.exp(-gamma * sq_dists) # GIP Kernel 相似性矩阵

    return gip_kernel