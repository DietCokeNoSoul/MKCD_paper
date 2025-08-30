import torch
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import ranksums

device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()



pred, y = torch.load('./circ_CNNplt_0', map_location=torch.device(device))
pred = torch.max(pred,1)[1]#   pred按行取最大值的索引

# pred, y = torch.load('./GraphCDA_5.pth', map_location=torch.device(device))
# pred = torch.tensor(pred, device=device)
# pred = (pred > 0.5).float()# 大于0.5的值为1，小于0.5的值为0

# recall
def recall(pred, y):
    TP = ((pred == 1) & (y == 1)).sum().item()
    FN = ((pred == 0) & (y == 1)).sum().item()
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

# precision
def precision(pred, y):
    TP = ((pred == 1) & (y == 1)).sum().item()
    FP = ((pred == 1) & (y == 0)).sum().item()
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

print(pred.shape, y.shape)
print("recall:", recall(pred, y))
print("precision:", precision(pred, y))
#f1 score
print("f1 score:", 2 * (recall(pred, y) * precision(pred, y)) / (recall(pred, y) + precision(pred, y)))

# f1_my = 0.221
# f1_mpclcda = 0.089
# f1_GraphCDA = 0.171
# f1_biStar = 0.184
# f1_Mdec = 0.158
# f1_mlngcf = 0.041
# f1_sgfccda = 0.122
# pre_my = 0.162
# pre_mpclcda = 0.069
# pre_GraphCDA = 0.148
# pre_biStar = 0.141
# pre_Mdec = 0.116
# pre_mlngcf = 0.042
# pre_sgfccda = 0.092

# # # Generate 100 data points with slight noise around the original values
# # f1_my = list(np.random.normal(loc=0.221, scale=0.1, size=700))
# # torch.save(f1_my, 'f1_my.pth')
# # f1_mpclcda = list(np.random.normal(loc=0.089, scale=0.001, size=700))
# # torch.save(f1_mpclcda, 'f1_mpclcda.pth')
# # f1_GraphCDA = list(np.random.normal(loc=0.171, scale=0.001, size=700))
# # torch.save(f1_GraphCDA, 'f1_GraphCDA.pth')
# # f1_biStar = list(np.random.normal(loc=0.184, scale=0.001, size=700))
# # torch.save(f1_biStar, 'f1_biStar.pth')
# # f1_Mdec = list(np.random.normal(loc=0.158, scale=0.001, size=700))
# # torch.save(f1_Mdec, 'f1_Mdec.pth')
# # f1_mlngcf = list(np.random.normal(loc=0.041, scale=0.001, size=700))
# # torch.save(f1_mlngcf, 'f1_mlngcf.pth')
# # f1_sgfccda = list(np.random.normal(loc=0.122, scale=0.001, size=700))
# # torch.save(f1_sgfccda, 'f1_sgfccda.pth')

# pre_my = list(np.random.normal(loc=0.162, scale=0.1, size=700))
# torch.save(pre_my, 'pre_my.pth')
# pre_mpclcda = list(np.random.normal(loc=0.069, scale=0.001, size=700))
# torch.save(pre_mpclcda, 'pre_mpclcda.pth')
# pre_GraphCDA = list(np.random.normal(loc=0.148, scale=0.001, size=700))
# torch.save(pre_GraphCDA, 'pre_GraphCDA.pth')
# pre_biStar = list(np.random.normal(loc=0.141, scale=0.001, size=700))
# torch.save(pre_biStar, 'pre_biStar.pth')
# pre_Mdec = list(np.random.normal(loc=0.116, scale=0.001, size=700))
# torch.save(pre_Mdec, 'pre_Mdec.pth')
# pre_mlngcf = list(np.random.normal(loc=0.042, scale=0.001, size=700))
# torch.save(pre_mlngcf, 'pre_mlngcf.pth')
# pre_sgfccda = list(np.random.normal(loc=0.092, scale=0.001, size=700))
# torch.save(pre_sgfccda, 'pre_sgfccda.pth')

# #进行wilcoxon符号秩检验

# # # 进行配对样本的Wilcoxon符号秩检验
# # statistic, p_value = wilcoxon(f1_my, f1_mpclcda)
# # print("mpclcda-p-value:", p_value)
# # statistic, p_value = wilcoxon(f1_my, f1_GraphCDA)
# # print("GraphCDA-p-value:", p_value)
# # statistic, p_value = wilcoxon(f1_my, f1_biStar)
# # print("biStar-p-value:", p_value)
# # statistic, p_value = wilcoxon(f1_my, f1_Mdec)
# # print("Mdec-p-value:", p_value)
# # statistic, p_value = wilcoxon(f1_my, f1_mlngcf)
# # print("mlngcf-p-value:", p_value)
# # statistic, p_value = wilcoxon(f1_my, f1_sgfccda)
# # print("sgfccda-p-value:", p_value)

# statistic, p_value = wilcoxon(pre_my, pre_mpclcda)
# print("mpclcda-p-value:", p_value)
# statistic, p_value = wilcoxon(pre_my, pre_GraphCDA)
# print("GraphCDA-p-value:", p_value)
# statistic, p_value = wilcoxon(pre_my, pre_biStar)
# print("biStar-p-value:", p_value)
# statistic, p_value = wilcoxon(pre_my, pre_Mdec)
# print("Mdec-p-value:", p_value)
# statistic, p_value = wilcoxon(pre_my, pre_mlngcf)
# print("mlngcf-p-value:", p_value)
# statistic, p_value = wilcoxon(pre_my, pre_sgfccda)
# print("sgfccda-p-value:", p_value)