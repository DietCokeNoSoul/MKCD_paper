import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import tqdm

n = 5
_, ds, _, _, tei = torch.load('circ_CNN.pth')

def caculate_TPR_FPR(RD, B):
    f = np.zeros(shape=(RD.shape[0], 1))
    for i in range(RD.shape[0]):
        f[i] = np.sum(RD[i] > (-1))

    # 关联矩阵
    old_id = np.argsort(-RD)  # 记录排序前的位置
    min_f = int(np.min(f))  # 最小的有效数据的数目
    max_f = int(np.max(f))  # 最大的有效数据的数目
    TP_FN = np.zeros((RD.shape[0], 1), dtype=np.float64)  # 真正例总数
    FP_TN = np.zeros((RD.shape[0], 1), dtype=np.float64)  # 真反例总数
    TP = np.zeros((RD.shape[0], max_f), dtype=np.float64)  # 正例被判断为正例的个数
    TP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    FP = np.zeros((RD.shape[0], max_f), dtype=np.float64)  # 假例被判断为正例的个数
    FP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    P = np.zeros((RD.shape[0], max_f), dtype=np.float64)  # 查准率
    P2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)  # 查准率

    for i in range(RD.shape[0]):
        TP_FN[i] = sum(B[i] == 1)  # 每行中求1的个数，包含TP（真正类）,FN（假负类）
        FP_TN[i] = sum(B[i] == 0)  # 每行中求0的个数，包含FP(假正类)，TN（真负类）

    for i in range(RD.shape[0]):  # 遍历RD的每行
        kk = f[i] / min_f  # 这行数据的有效数据数目是最小数目的多少倍

        for j in range(int(f[i].item())):  # 遍历每行A的有效列数
            if j == 0:  # 如果这行的有效列数为0，则按下面的算
                if B[i][old_id[i][j]] == 1:  # j*kk+(kk-1) 按比例抽样
                    FP[i][j] = 0
                    TP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = 0
                    FP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
            else:  # 如果这行的有效列数不是0，则按下面的算
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = FP[i][j - 1]
                    TP[i][j] = TP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = TP[i][j - 1]
                    FP[i][j] = FP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)

    ki = 0  # 无效行的数目
    for i in range(RD.shape[0]):
        if TP_FN[i] == 0:
            TP[i] = 0
            FP[i] = 0
            ki = ki + 1
        else:
            TP[i] = TP[i] / TP_FN[i]  # 此时的TP里其实装的是TPR
            FP[i] = FP[i] / FP_TN[i]

    for i in range(RD.shape[0]):
        kk = f[i] / min_f  # 这行数据的有效数据数目是最小数目的多少倍
        for j in range(min_f):
            TP2[i][j] = TP[i][int(np.round(((j + 1) * kk)).item()) - 1]
            FP2[i][j] = FP[i][int(np.round(((j + 1) * kk)).item()) - 1]
            P2[i][j] = P[i][int(np.round(((j + 1) * kk)).item()) - 1]

    TPR = TP2.sum(0) / (TP.shape[0] - ki)

    FPR = FP2.sum(0) / (FP.shape[0] - ki)

    P = P2.sum(0) / (P.shape[0] - ki)
    return TPR, FPR, P


def fold_5(TPR, FPR, PR, fold, min):
    F_TPR = np.zeros((fold, min))
    F_FPR = np.zeros((fold, min))
    F_P = np.zeros((fold, min))
    for i in range(fold):
        k = len(TPR[i]) / min  # 这行数据的有效数据数目是最小数目的多少倍
        for j in range(min):
            F_TPR[i][j] = TPR[i][int(np.round(((j + 1) * k)).item()) - 1]
            F_FPR[i][j] = FPR[i][int(np.round(((j + 1) * k)).item()) - 1]
            F_P[i][j] = PR[i][int(np.round(((j + 1) * k)).item()) - 1]
    TPR_5 = F_TPR.sum(0) / (fold)
    FPR_5 = F_FPR.sum(0) / (fold)
    PR_5 = F_P.sum(0) / (fold)
    return TPR_5, FPR_5, PR_5


def curve(TPR, FPR, P):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title("ROC curve  (AUC = %.4f)" % (metrics.auc(FPR, TPR)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(FPR, TPR)
    plt.subplot(122)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title('PR curve (APUC = %.4f)' % (metrics.auc(TPR, P) + (TPR[0] * P[0])))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.plot(TPR, P)
    plt.show()


def auc_pr_5(n, pred_list_1, y_list_1):
    MIN = 99999

    TPR_ALL, FPR_ALL, P_ALL = [], [], []
    for cros in range(5):
        pred = pred_list_1[cros] # 113315
        y = y_list_1[cros] # 113315
        # # 得到关联概率
        # pred = torch.softmax(pred, dim=1)[:, 1]

        # 全为-1的关联矩阵大小相同的张量
        trh = torch.zeros(ds.shape[0], ds.shape[1]) - 1
        tlh = torch.zeros(ds.shape[0], ds.shape[1]) - 1

        # 测试集存储的是索引 第0行表示行索引,第1行表示列索引
        trh[tei[cros][0], tei[cros][1]] = pred
        tlh[tei[cros][0], tei[cros][1]] = y

        trh = trh.numpy()
        tlh = tlh.numpy()

        TPR, FPR, P = caculate_TPR_FPR(trh, tlh)

        MIN = min(MIN, int(len(TPR)))

        TPR_ALL.append(TPR)
        FPR_ALL.append(FPR)
        P_ALL.append(P)

    TPR, FPR, P = fold_5(TPR_ALL, FPR_ALL, P_ALL, n, MIN)
    return TPR, FPR, P


pred_list_1 = []
y_list_1 = []
pred_list_2 = []
y_list_2 = []
# for i in range(5):
#     pred = torch.load(rf'mycode\\testData\\SGFCCDA\\SGFCCDAoutput{i}').cpu().squeeze(1) # 113315
#     y = torch.load(rf'mycode\\testData\\SGFCCDA\\SGFCCDAlabel{i}').cpu().squeeze(1) # 113315
#     pred_list_1.append(pred)
#     y_list_1.append(y)
for i in range(5):
    pred, y = torch.load(rf'circ_CNNplt_{i}', map_location=torch.device('cpu'))
    pred = pred[:, 1]
    pred_list_2.append(pred)
    y_list_2.append(y)
# TPR_1, FPR_1, P_1 = auc_pr_5(5, pred_list_1, y_list_1)
TPR_2, FPR_2, P_2 = auc_pr_5(5, pred_list_2, y_list_2)
# curve(TPR_1, FPR_1, P_1)
curve(TPR_2, FPR_2, P_2)