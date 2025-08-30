import torch
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def Cal_ROC_PR(y_true, y_scores):
    # 对预测概率进行排序
    y_scores, sorted_indices = torch.sort(y_scores, dim=0, descending=False)
    y_true = y_true[sorted_indices]

    # 初始化TPR和FPR，Precision和Recall
    TPR_list, FPR_list, Precision_list, Recall_list = [], [], [], []

    # 计算
    for threshold in (y_scores):
        y_rescore = y_scores >= threshold
        True_Positive = torch.sum(torch.logical_and(y_rescore, y_true)).float()
        True_Negative = torch.sum(torch.logical_and(~y_rescore, ~y_true)).float()
        False_Positive = torch.sum(torch.logical_and(y_rescore, ~y_true)).float()
        False_Negative = torch.sum(torch.logical_and(~y_rescore, y_true)).float()


        TPR = True_Positive / (True_Positive + False_Negative) if True_Positive > 0 else torch.tensor(0.).float()
        FPR = False_Positive / (False_Positive + True_Negative) if False_Positive > 0 else torch.tensor(0.).float()
        Precision = True_Positive / (True_Positive + False_Positive) if True_Positive > 0 else torch.tensor(0.).float()

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())
        Precision_list.append(Precision.item())
    Recall_list = TPR_list

    # AUC为ROC曲线下方面积，AUCP为PR曲线下方面积
    FPR_temp, TPR_temp = zip(*sorted(list(zip(FPR_list, TPR_list)), key=lambda x: x[0], reverse=False))
    FPR_list, TPR_list = list(FPR_temp), list(TPR_temp)
    AUC_ROC = np.trapz(TPR_list, FPR_list)

    Recall_temp, Precision_temp = zip(*sorted(list(zip(Recall_list, Precision_list)), key=lambda x: x[0], reverse=False))
    Recall_list, Precision_list = list(Recall_temp), list(Precision_temp)
    AUC_PR = np.trapz(Precision_list, Recall_list)

    return FPR_list, TPR_list, Precision_list, Recall_list, AUC_ROC, AUC_PR


def Draw_ROC_PR(y_test, y_score):
    # 定义颜色列表，用于区分不同的 ROC 曲线
    light_colors = [
        '#FFCCCC',  # 浅红色
        '#CCFFCC',  # 浅绿色
        '#CCCCFF',  # 浅蓝色
        '#FFCC99',  # 浅橙色
        '#CC99FF',  # 浅紫色
        '#FF99CC',  # 浅粉红色
        '#C2C2F0',  # 浅灰色
        '#99CC00',  # 浅橄榄绿
        '#00CC99',  # 浅青色
        '#CCCCCC',  # 浅灰色
    ]

    plt.figure(figsize=(12, 5))
    for i in range(len(y_test)):
        FPR_list, TPR_list, Precision_list, Recall_list, A_ROC, A_PR = Cal_ROC_PR(y_test[i], y_score[i])

        graph_lists = {
            'FPR_list': FPR_list,
            'TPR_list': TPR_list,
            'Precision_list': Precision_list,
            'Recall_list': Recall_list,
            'A_ROC': A_ROC,
            'A_PR': A_PR,
        }
        torch.save(graph_lists, f'graph_lists_{i}.pt')


        plt.subplot(1, 2, 1)
        plt.plot(FPR_list, TPR_list,
                 color=light_colors[2 * i % len(light_colors)], lw=1, label=f'fold {i + 1} ({A_ROC:.4f})')
        plt.subplot(1, 2, 2)
        plt.plot(Recall_list, Precision_list,
                 color=light_colors[(2 * i + 1) % len(light_colors)], lw=1, label=f'fold {i + 1} ({A_PR:.2f})')

    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()
    plt.savefig('roc_pr_curve.png', bbox_inches='tight')

