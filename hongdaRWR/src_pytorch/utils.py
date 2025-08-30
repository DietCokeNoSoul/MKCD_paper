from numpy import divide, concatenate, exp, log, power,floor_divide, true_divide,ndindex, argwhere, array_split, ndarray, eye, hstack, vstack, stack, matmul, diag_indices_from,  repeat, load, save, array, newaxis
from numpy.random import shuffle
from os import makedirs
from os.path import exists, join

from torch import device, save as save_torch
from torch import load as load_torch
from torch.utils.data.dataset import Dataset
from torch.nn import Module


def partition(a: ndarray, index: int, total: int) -> list:
    '''将列表a按照顺序等距分为total段，并返回其中第index段元素构成的列表。
    '''
    return array_split(a, total)[index]


def remaining_partition(a: ndarray, index: int, total: int) -> list:
    '''将列表a按照顺序等距分为total段，并返回除第index段外的所有元素构成的列表。
    '''
    result = [each for i, each in enumerate(
        array_split(a, total)) if i != index]
    return concatenate(result)


def folds_of_cross_validation(a: ndarray, folds: int):
    ''''''
    groups = array_split(a, folds)
    splits = []
    for i in range(folds):
        index = filter(lambda x: x != i, range(folds))
        x = concatenate([groups[idx] for idx in index])
        y = groups[i]
        splits.append({
            "train": x,
            "test": y
        })
    return splits


def positive_pairs(RD: ndarray):
    '''返回矩阵RD中值为1的所有元素坐标(x,y)构成的列表
    '''
    positives = argwhere(RD == 1)
    return positives


def negative_pairs(RD: ndarray):
    '''返回矩阵RD中值为0的所有元素坐标(x,y)构成的列表
    '''
    negatives = argwhere(RD == 0)
    return negatives


def reset_associations(A: ndarray, pairs: ndarray):
    '''返回矩阵A的副本A'，并将坐标包含在pairs的元素置零'''
    B = A.copy()
    for x, y in pairs:
        B[x, y] = 0
    return B


def get_positives_negatives(A: ndarray):
    '''分别返回关系A中的正样本和负样本坐标,同时打乱顺序'''
    positives = positive_pairs(A)
    negatives = negative_pairs(A)
    shuffle(positives)
    shuffle(negatives)
    return positives, negatives


def divid_pairs(positives: list, negatives: list, index: int, groups: int):
    '''分别将positives和negatives均匀分为groups组，返回第index组，打乱后，放到同一列表中返回
    '''
    pos = partition(positives, index, groups)
    neg = partition(negatives, index, groups)
    res = concatenate([pos, neg])
    shuffle(res)
    return res


def remaining_pairs(positives: list, negatives: list, index: int, groups: int):
    '''分别将positives和negatives均匀分为groups组，返回除第index组外所有元素构成的列表，同时打乱顺序'''

    pos = remaining_partition(positives, index, groups)
    neg = remaining_partition(negatives, index, groups)
    res = concatenate([pos, neg])
    shuffle(res)
    return res


def list_sub(a: ndarray, b: ndarray):
    '''集合实现的列表差集'''
    a = a.tolist()
    b = b.tolist()
    a = [tuple(each) for each in a]
    b = [tuple(each) for each in b]
    a = set(a)
    b = set(b)
    return array(list(a-b))


prefix = './data/cross_validation'


def save_dataset(fold: int, dataset, file_name: str):
    path = fold_path(fold)
    save_torch(dataset, join(path, file_name))


def load_dataset(fold: int, file_name):
    path = fold_path(fold)
    return load_torch(join(path, file_name))


def save_pairs(pairs: ndarray, file_name: str):
    '''将pairs保存为prefix目录下的file_name文件'''
    save(file_name, pairs)


def load_pairs(file_name: str):
    '''从prefix目录下的file_name加载pairs'''
    return load(file_name)


def fold_path(fold: int):
    '''获得多倍交叉的第fold交叉的数据路径'''
    path = join(prefix, str(fold))
    if not exists(path):
        makedirs(path)
    return path


def save_cross_validation(fold: int, trains: ndarray, validates: ndarray, RD: ndarray):
    '''保存多倍交叉的验证数据索引

    Args:
        fold:第fold交叉
        trains:用于训练的药物疾病对
        validates:用于验证是否过拟合的药物疾病对
        RD: 药物疾病对原始关联标签
    '''
    path = fold_path(fold)
    pairs = array(ndindex(RD.shape))
    tests = list_sub(pairs, trains)
    save_pairs(trains, join(path, 'train_indexs.npy'))
    save_pairs(tests, join(path, 'test_indexs.npy'))
    save_pairs(validates, join(path, 'validate_indexs.npy'))
    save(join(path, 'RD.npy'), RD)


def load_cross_train(fold: int):
    '''加载第fold交叉的训练样本和置零关系

    Returns:
        train:用于训练的药物疾病对
        reset:需要置零的、假设为未知的药物疾病关系
    '''
    path = fold_path(fold)
    train = load(join(path, 'train_indexs.npy'))
    validate = load(join(path, 'validate_indexs.npy'))
    return train, validate


def load_cross_test(fold: int):
    '''返回第fold交叉的测试样本'''
    path = fold_path(fold)
    test = load(join(path, 'test_indexs.npy'))
    return test


def save_model(fold: int, model: Module, name: str):
    path = fold_path(fold)
    save_torch(model.state_dict(), join(path, name))


def load_model(fold: int, model: Module, name: str):
    path = fold_path(fold)
    states = load_torch(join(path, name))
    model.load_state_dict(states)


def save_models(fold: int, models: list, names: list):
    '''以列表names中的名字保存列表models中的模型'''
    path = fold_path(fold)
    for model, name in zip(models, names):
        if isinstance(model, Dataset):
            save_torch(model.to_device(device("cpu")), join(path, name))
        elif isinstance(model, Module):
            save_torch(model.cpu(), join(path, name))
        else:
            save_torch(model, join(path, name))


def load_models(fold: int, names: list):
    '''从列表names中加载第fold交叉中的模型'''
    path = fold_path(fold)
    ms = list(load_torch(join(path, name)) for name in names)
    return ms


def save_scores(fold: int, scores: ndarray):
    '''保存第fold交叉下的得分到scores.npy中'''
    path = fold_path(fold)
    save(join(path, 'scores.npy'), scores)


def load_scores(fold: int):
    '''从第fold交叉中加载得分（scores.npy）'''
    path = fold_path(fold)
    return load(join(path, 'scores.npy'))


def save_labels(fold: int, labels: ndarray):
    '''保存第fold交叉下的标签到labels.npy中'''
    path = fold_path(fold)
    save(join(path, 'labels.npy'), labels)


def load_labels(fold: int):
    '''从第fold交叉中加载标签（labels.npy）'''
    path = fold_path(fold)
    return load(join(path, 'labels.npy'))


def line_normalization(a: ndarray) -> ndarray:
    '''行归一化'''
    an = a.copy()
    s = a.sum(axis=1,keepdims=True)
    s[s==0]=1
    return true_divide(an,s)

def softmax(a:ndarray,t:float)->ndarray:
    X=a.copy()
    for i in range(X.shape[0]):
        X[i]=exp(divide(X[i],t))
        X[i]=divide(X[i],X[i].sum())
    return X


def rand_walk(alpha: float, steps: int, A: ndarray) -> list:
    '''以重启概率α自初始状态随机游走RD共step步，返回含初始状态的关系序列列表'''
    # A=A.copy()
    # A[diag_indices_from(A)]=0
    # P=line_normalization(A)
    # S=P
    # ans=[S]
    # W=S.copy()
    # for s in range(steps):
    #     W=alpha*matmul(P,W)+(1-alpha)*S
    #     ans.append((W*10000))
    P=A.copy()
    P[diag_indices_from(P)]=0
    P=line_normalization(P)
    # E=eye(A.shape[0])
    E=A
    ans=[E]
    W=E
    for s in range(steps):
        W=(1-alpha)*matmul(P.T,W)+alpha*E
        ans.append(W.T)
    return ans


def joint(RRs: list, DDs: list, RD: ndarray) -> ndarray:
    '''根据从RRS和DDs中选取不同的相似性构成双层网'''
    joint_matrixs = []
    for RR in RRs:
        for DD in DDs:
            m = vstack((
                hstack((RR, RD)),
                hstack((RD.T, DD))
            ))
            joint_matrixs.append(m)
    return stack(joint_matrixs)


def take_simple(simples: ndarray, num: int) -> ndarray:
    '''从simples中等距采样num个'''
    length = simples.shape[0]
    if length == num:
        return simples
    rate = length/num
    index = 0
    res = []
    while index < length-1:
        res.append(simples[int(index)])
        index += rate
    return vstack(res)


def simples_with_positive(score_label: ndarray) -> ndarray:
    '''选择score_label中包含正样本的行，构成新的得分-标签张量'''
    length = score_label.shape[0]
    _length = score_label.shape[1]
    selected = [False for i in range(length)]
    for i in range(length):
        for j in range(_length):
            if score_label[i, j, 1] == 1:
                selected[i] = True
                break
    return array([a for a, select in zip(score_label, selected) if select])
