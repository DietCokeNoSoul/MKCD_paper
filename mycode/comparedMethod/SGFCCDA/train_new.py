import argparse
import math
import random
import time
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
parser.add_argument('--epochs', type=int, default=10)
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

    #circRNA-disease5*834*138 覆盖过得特征矩阵5*1527*1527 训练集索引表5*2*1576 测试集索引表5*2*113315
    _, cd, features, trainSet_index, testSet_index = torch.load('circ_CNN.pth') 
    rna_numbers = cd.shape[0] # 834
    dis_number = cd.shape[1] # 138

    # for i in range(5):
    i = 4
    circSimi_mat = features[i][:834, :834].to(torch.float32) # 834*834 circ相似矩阵
    disSimi_mat = features[i][834:972, 834:972].to(torch.float32) # 138*138 dis相似矩阵
    new_circrna_disease_matrix = cd # 834*138 circRNA-disease关联矩阵

    Adj_circ = (circSimi_mat > 0.3).int() # 834*834 circRNA邻接矩阵
    Adj_dis = (disSimi_mat > 0.3).int() # 138*138 disease邻接矩阵
    adj_1 = np.hstack((Adj_circ, new_circrna_disease_matrix)) # 834*972
    adj_2 = np.hstack((new_circrna_disease_matrix.T, Adj_dis)) # 138*972
    adj = torch.tensor(np.vstack((adj_1, adj_2))).to(device) # 972*972

    circSimi_mat = circSimi_mat.to(device)
    disSimi_mat = disSimi_mat.to(device)
    new_circrna_disease_matrix = new_circrna_disease_matrix.to(device)
    
    model = Model(834, 138, 128, args.hidden_size, args.num_layers, args.dropout,
                    args.drop_rate, args.embedding_size, 2).to(device)
    # 声明参数优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 模型训练
    model.train()
    for epoch in range(args.epochs):
        tarin_start_time = time.time()
        # 输入特征矩阵adj（972*972），circRNA相似度矩阵（834*834），disease相似度矩阵（138*138），circRNA-disease关联矩阵（834*138）
        train_predict_result, train_lable = model(adj, circSimi_mat, disSimi_mat, new_circrna_disease_matrix, train_model=True, device = device,trainSet_index=trainSet_index[i])
        loss = F.binary_cross_entropy(train_predict_result, train_lable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_end_time = time.time()
        print('Epoch: %d Time: %.4f' % (epoch + 1, train_end_time - tarin_start_time))
        print('Epoch: %d | Loss: %.4f' % (epoch + 1, loss.item()))

    # 模型评估
    test_start_time = time.time()
    model.eval() 
    with torch.no_grad():
        test_predict_result, test_lable = model(adj, circSimi_mat, disSimi_mat, new_circrna_disease_matrix, train_model=False, device = device, trainSet_index=testSet_index[i])
    predicted_labels = (test_predict_result >= 0.5).float()  # 将预测结果转换为0或1
    correct_predictions = (predicted_labels == test_lable).float().sum()
    accuracy = correct_predictions / test_lable.size(0)
    print('Test Accuracy: %.4f' % accuracy.item())
    test_end_time = time.time()
    print('Test Time: %.4f' % (test_end_time - test_start_time))
    torch.save(test_predict_result,RF'SGFCCDAoutput{i}')
    torch.save(test_lable,RF'SGFCCDAlabel{i}') 