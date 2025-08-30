import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_features_choose(rel_adj_mat, features_embedding, trainSet_index, split):
    rna_nums = rel_adj_mat.size()[0]
    features_embedding_rna = features_embedding[0:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:features_embedding.size()[0], :]
    train_features_input, train_lable = [], []

    for idx in range(trainSet_index.shape[1]):
        r, d = trainSet_index[0, idx], trainSet_index[1, idx]
        train_features_input.append((features_embedding_rna[r, :] * features_embedding_dis[d, :]).unsqueeze(0))
        if idx < split:
            train_lable.append(1)
        else:
            train_lable.append(0)

    train_features_input = torch.cat(train_features_input, dim=0)
    train_lable = torch.FloatTensor(np.array(train_lable)).unsqueeze(1)
    return train_features_input.to(device), train_lable.to(device)


def test_features_choose(rel_adj_mat, features_embedding):
    rna_nums, dis_nums = rel_adj_mat.size()[0], rel_adj_mat.size()[1]
    features_embedding_rna = features_embedding[0:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:features_embedding.size()[0], :]
    test_features_input, test_lable = [], []

    for i in range(rna_nums):
        for j in range(dis_nums):
            test_features_input.append((features_embedding_rna[i, :] * features_embedding_dis[j, :]).unsqueeze(0))
            test_lable.append(rel_adj_mat[i, j])

    test_features_input = torch.cat(test_features_input, dim=0)
    test_lable = torch.FloatTensor(np.array(test_lable)).unsqueeze(1)
    return test_features_input.to(device), test_lable.to(device)


def sort_matrix(score_matrix, interact_matrix):
    '''
    实现矩阵的列元素从大到小排序
    1、np.argsort(data,axis=0)表示按列从小到大排序
    2、np.argsort(data,axis=1)表示按行从小到大排序
    '''
    sort_index = np.argsort(-score_matrix, axis=0)  # 沿着行向下(每列)的元素进行排序
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:, i] = score_matrix[:, i][sort_index[:, i]]
        y_sorted[:, i] = interact_matrix[:, i][sort_index[:, i]]
    return y_sorted, score_sorted


def draw_alone_validation_roc_line(tpr_arr_matrix, fpr_arr_matrix):
    # 保存每条曲线的对象
    handlist = []

    # 设置字体格式
    font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }

    plt.rcParams['font.family'] = 'Times New Roman'

    # 开启一个窗口，figsize设置窗口大小
    # figsize = 12, 9
    figsize = 6, 5
    figure, ax = plt.subplots(figsize=figsize)
    for row in range(tpr_arr_matrix.shape[0]):
        data_tpr, data_fpr = tpr_arr_matrix[row, :], fpr_arr_matrix[row, :]
        roc_auc = np.trapz(data_tpr, data_fpr)

        b, = plt.plot(data_fpr, data_tpr, label='ROC fold{} (AUC={})'.format(row + 1, round(roc_auc, 4)),
                      linewidth=2)
        handlist.append(b)

    plt.legend(handles=handlist, prop=font, loc='lower right')

    plt.title('ROC curves', font)
    plt.xlabel('False positive rate', font)
    plt.ylabel('True positive rate', font)

    # 设置坐标轴间隔
    x_major_locator = MultipleLocator(0.2)
    y_major_locator = MultipleLocator(0.2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # plt.show()  # 展示绘图
