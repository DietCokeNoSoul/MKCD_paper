import numpy as np
import itertools
import torch

circ_name = np.load("mycode\data\circ_disease\circRNA_name.npy")
disease_name = np.load("mycode\data\circ_disease\disease_name.npy")
# 创建一个834*138的空矩阵
pred = torch.zeros(834, 138).to("cuda:0")
pred_score = torch.zeros(834, 138).to("cuda:0")
y = torch.zeros(834, 138).to("cuda:0")
# 将预测结果和真实结果填入矩阵,test_index为2*113315，第一列是cRNA索引，第二列是疾病索引
for i in range(5):
    pred_s, y_t = torch.load('circ_CNNplt_%d' % i)
    # pred按行取最大值的索引
    pred_t = pred_s.argmax(dim=1).to("cuda:0")
    _, cd, _, _, testSet_index = torch.load('circ_CNN.pth')
    test_index = testSet_index[i].to("cuda:0")  # 2*113315
    pred[test_index[0], test_index[1]] = pred_t.float()
    # 只保留pred_score中的最大值
    pred_score[test_index[0], test_index[1]] = torch.max(pred_score[test_index[0], test_index[1]], pred_s[:, 1])
    y[test_index[0], test_index[1]] = y_t.float()

print(pred.sum(), y.sum())
# 计算每个疾病关联的circRNA数量
circRNA_counts = (pred > 0).sum(dim=0).cpu().numpy()
# 选出关联circRNA最多的3个疾病
top3_diseases = np.argsort(circRNA_counts)[-10:][::-1]

for disease_idx in top3_diseases:
    print(f"Disease: {disease_name[disease_idx]}, Associated circRNAs: {circRNA_counts[disease_idx]}")
    # 获取该疾病列的关联分数并排序
    scores = pred_score[:, disease_idx].cpu().numpy()
    top_circRNA_indices = np.argsort(scores)[-30:][::-1]
    top_circRNA_names = circ_name[top_circRNA_indices]
    # print(f"Top 15 circRNAs for Disease {disease_name[disease_idx]}:")
    torch.save([disease_name[disease_idx], top_circRNA_names], f"top_circRNAs_for_{disease_name[disease_idx]}.pkl")
    print(disease_name[disease_idx], top_circRNA_names)
    print()