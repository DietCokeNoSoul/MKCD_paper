import argparse
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from model import Model
from utils import sort_matrix, draw_alone_validation_roc_line
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1, help='--ScaleGCN')
parser.add_argument('--drop_rate', type=float, default=0.1, help='--MLP')
parser.add_argument('--embedding_size', type=float, default=1536)

device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    A = np.loadtxt('mycode\\comparedMethod\\SGFCCDA\\data\\CircR2Disease\\associationMatrix.csv', delimiter=',') # 585*88
    circSimi = np.loadtxt('mycode\\comparedMethod\\SGFCCDA\\data\\CircR2Disease\\circRNA_similarity.csv', delimiter=',') # 585*585
    disSimi = np.loadtxt('mycode\\comparedMethod\\SGFCCDA\\data\\CircR2Disease\\disease_similarity.csv', delimiter=',') # 88*88
    circSimi_mat = torch.from_numpy(circSimi).to(torch.float32) # 585*585
    disSimi_mat = torch.from_numpy(disSimi).to(torch.float32) # 88*88

    circrna_disease_matrix = np.copy(A) # 585*88
    rna_numbers = circrna_disease_matrix.shape[0] # 585
    dis_number = circrna_disease_matrix.shape[1] # 88

    positive_index_tuple = np.where(circrna_disease_matrix == 1)
    positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1])) # 650*2
    random.shuffle(positive_index_list)
    positive_split = math.ceil(len(positive_index_list) / 5) # 130

    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    count = 0

    for i in range(0, len(positive_index_list), positive_split):
        count = count + 1
        positive_train_index_to_zero = positive_index_list[i: i + positive_split] # 130*2
        new_circrna_disease_matrix = circrna_disease_matrix.copy() # 585*88

        for index in positive_train_index_to_zero:
            new_circrna_disease_matrix[index[0], index[1]] = 0

        new_circrna_disease_matrix_tensor = torch.from_numpy(new_circrna_disease_matrix).to(device) # 585*88

        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix

        Adj_circ = np.where(circSimi > 0.3, 1, 0) # 585*585
        Adj_dis = np.where(disSimi > 0.3, 1, 0) # 88*88
        adj_1 = np.hstack((Adj_circ, new_circrna_disease_matrix)) # 585*673 把circRNA的邻接矩阵和circRNA-disease矩阵拼接
        adj_2 = np.hstack((new_circrna_disease_matrix.transpose(), Adj_dis)) # 88*673 把circRNA-disease矩阵和disease的邻接矩阵拼接
        adj = torch.tensor(np.vstack((adj_1, adj_2))).to(device) # 673*673 adj_1和adj_2垂直拼接

        circSimi_mat = circSimi_mat.to(device)
        disSimi_mat = disSimi_mat.to(device)

        model = Model(585, 88, 128, args.hidden_size, args.num_layers, args.dropout,
                      args.drop_rate, args.embedding_size, 2).to(device)
        # 声明参数优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 模型训练
        model.train()
        for epoch in range(args.epochs):
            train_predict_result, train_lable = model(adj, circSimi_mat, disSimi_mat, new_circrna_disease_matrix_tensor,
                                                      train_model=True,device = device) # 输入特征矩阵adj（673*673），circRNA相似度矩阵（585*585），disease相似度矩阵（88*88），circRNA-disease关联矩阵（585*88）
            loss = F.binary_cross_entropy(train_predict_result, train_lable)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: %d | Loss: %.4f' % (epoch + 1, loss.item()))

        # 模型评估
        model.eval()
        with torch.no_grad():
            test_predict_result, test_lable = model(adj, circSimi_mat, disSimi_mat, new_circrna_disease_matrix_tensor,
                                                    train_model=False)
        prediction_matrix = np.zeros(circrna_disease_matrix.shape)
        for num in range(test_predict_result.size()[0]):
            row_num = num // dis_number
            col_num = num % dis_number
            prediction_matrix[row_num, col_num] = test_predict_result[num, 0]

        zero_matrix = np.zeros(prediction_matrix.shape).astype('int64')
        prediction_matrix_temp = prediction_matrix.copy()
        prediction_matrix_temp = prediction_matrix_temp + zero_matrix
        min_value = np.min(prediction_matrix_temp)

        index_where_2 = np.where(roc_circrna_disease_matrix == 2)

        prediction_matrix_temp[index_where_2] = min_value - 20

        sorted_rna_dis_matrix, sorted_prediction_matrix = sort_matrix(prediction_matrix_temp,
                                                                      roc_circrna_disease_matrix)

        tpr_list = []
        fpr_list = []
        recall_list = []
        precision_list = []
        accuracy_list = []
        F1_list = []
        for cutoff in range(sorted_rna_dis_matrix.shape[0]):
            P_matrix = sorted_rna_dis_matrix[0:cutoff + 1, :]
            N_matrix = sorted_rna_dis_matrix[cutoff + 1:sorted_rna_dis_matrix.shape[0] + 1, :]
            TP = np.sum(P_matrix == 1)
            FP = np.sum(P_matrix == 0)
            TN = np.sum(N_matrix == 0)
            FN = np.sum(N_matrix == 1)
            tpr = TP / (TP + FN)
            fpr = FP / (FP + TN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            recall_list.append(recall)
            precision_list.append(precision)
            accuracy = (TN + TP) / (TN + TP + FN + FP)
            F1 = (2 * TP) / (2 * TP + FP + FN)
            F1_list.append(F1)
            accuracy_list.append(accuracy)

        index_accuracy = np.mean(accuracy_list)
        index_recall = np.mean(recall_list)
        index_precision = np.mean(precision_list)
        index_F1 = np.mean(F1_list)

        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr_list, tpr_list)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T
        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

        all_tpr.append(tpr_list)
        all_fpr.append(fpr_list)
        all_recall.append(recall_list)
        all_precision.append(precision_list)
        all_accuracy.append(accuracy_list)
        all_F1.append(F1_list)

    tpr_arr = np.array(all_tpr)
    fpr_arr = np.array(all_fpr)
    recall_arr = np.array(all_recall)
    precision_arr = np.array(all_precision)
    accuracy_arr = np.array(all_accuracy)
    F1_arr = np.array(all_F1)

    draw_alone_validation_roc_line(tpr_arr, fpr_arr)

    mean_cross_tpr = np.mean(tpr_arr, axis=0)
    mean_cross_fpr = np.mean(fpr_arr, axis=0)
    mean_cross_recall = np.mean(recall_arr, axis=0)
    mean_cross_precision = np.mean(precision_arr, axis=0)

    mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
    mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
    mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
    mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)

    roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
    AUPR = np.trapz(mean_cross_precision, mean_cross_recall)

    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC (AUC=%0.4f)' % roc_auc, linewidth=2, color='black')
    plt.legend(loc='lower right')
