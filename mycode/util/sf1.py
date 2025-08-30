import numpy as np
import itertools
import torch
import pandas as pd
import openpyxl

circ_name = np.load("mycode\data\circ_disease\circRNA_name.npy")
disease_name = np.load("mycode\data\circ_disease\disease_name.npy")
# 创建一个834*138的空矩阵
pred = torch.zeros(834, 138).to("cuda:0")
_, cd, _, _, testSet_index = torch.load('circ_CNN.pth')
testSet_index = testSet_index[0]
pred_p_1, y_p = torch.load('./circ_CNNplt_0', map_location=torch.device("cuda:0"))
pred_p_1 = torch.nn.functional.softmax(pred_p_1, dim=1)
pred_p_1 = pred_p_1[:, 1].to("cuda:0")
pred_p_2, y_p = torch.load('./circ_CNNplt_1', map_location=torch.device("cuda:0"))
pred_p_2 = torch.nn.functional.softmax(pred_p_2, dim=1)
pred_p_2 = pred_p_2[:, 1].to("cuda:0")
pred_p_3, y_p = torch.load('./circ_CNNplt_2', map_location=torch.device("cuda:0"))
pred_p_3 = torch.nn.functional.softmax(pred_p_3, dim=1)
pred_p_3 = pred_p_3[:, 1].to("cuda:0")
pred_p_4, y_p = torch.load('./circ_CNNplt_3', map_location=torch.device("cuda:0"))
pred_p_4 = torch.nn.functional.softmax(pred_p_4, dim=1)
pred_p_4 = pred_p_4[:, 1].to("cuda:0")
pred_p_5, y_p = torch.load('./circ_CNNplt_4', map_location=torch.device("cuda:0"))
pred_p_5 = torch.nn.functional.softmax(pred_p_5, dim=1)
pred_p_5 = pred_p_5[:, 1].to("cuda:0")
# #pred_p取五个模型中的最大值
# pred_p = torch.stack([pred_p_1, pred_p_2, pred_p_3, pred_p_4, pred_p_5], dim=1)
# pred_p = torch.max(pred_p, dim=1).values

# pred_p取五个模型的平均值
pred_p = (pred_p_1 + pred_p_2 + pred_p_3 + pred_p_4 + pred_p_5) / 5

# # 对pred_p的两列做softmax
# pred_p = torch.nn.functional.softmax(pred_p, dim=1)
# pred_p = pred_p[:, 1].to("cuda:0")

# 将pred_p放回pred, 按照索引test_index（2*113315），第一列是cRNA索引，第二列是疾病索引，将pred_p的第一个元素在pred的索引为test_index的第一列
for i in range(len(testSet_index[0])):
    pred[testSet_index[0][i], testSet_index[1][i]] = pred_p[i]

# 创建一个空的DataFrame来保存结果
results = []

# 挑选出每个疾病关联的circRNA得分最高的15个circRNA
for disease_idx in range(pred.shape[1]):
    top_indices = torch.topk(pred[:, disease_idx], 15).indices
    for rank, circ_idx in enumerate(top_indices):
        results.append([
            disease_name[disease_idx],
            circ_name[circ_idx],
            rank + 1,
            pred[circ_idx, disease_idx].item()
        ])

# 将结果转换为DataFrame
df = pd.DataFrame(results, columns=["Disease name", "CircRNA name", "Rank", "Association score"])

# 保存为xls文件
df.to_excel("circRNA_disease_association.xlsx", index=False)
