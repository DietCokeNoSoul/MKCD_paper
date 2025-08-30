'''
任务:采用CNN来对circRNA-disease数据进行预测,并通过无责交叉验证,然后输出AUC和AUPR
数据集:circRNA_disease、circRNA_miRNA、disease_disease、disease_miRNA ||| circRNA_name、disease_name、miRNA_name
'''


# 导入第三方包
import sys
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

# 导入自定义包
import data_loader
import CNN_Model
from data_loader import MyDataset
from util import plt
from train_function import train


torch.cuda.empty_cache()
device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

'''
    数据加载与预处理
'''
# 读取数据 circRNA-disease5*834*834 覆盖过得特征矩阵5*834*138 训练集索引表5*2*1576 测试集索引表5*2*113315
cd, features, trainSet_index, testSet_index = torch.load('circ_drug_CNN_1_1_util.pth') 
print(cd.shape, features[0].shape, trainSet_index[0].shape, testSet_index[0].shape)

'''
    超参数设置
'''
learn_rate=0.0005   #学习率
epoch=10            #伦次
batch=32            #批大小


'''
    训练模型
'''       
for times in range(3):
    for i in range(5): #5折交叉验证
        print('cross:%d'%(i+1))
        net=CNN_Model.Net().to(device)  #放入cuda
        cost=nn.CrossEntropyLoss()      #交叉熵损失
        #优化器  Adam   参数（模型参数，学习率，l2正则化）
        optimizer=torch.optim.Adam(net.parameters(),learn_rate,weight_decay=0.00005)
        # 将1576个（x1,y1,label)按照32大小分割，每一个batch格式（32,x1,x2,label）,(x1,x2)是cd索引,对应一对circRNA与disease,label是cd在(x1,x2)的数值
        trainSet=DataLoader(MyDataset(trainSet_index[i],cd),batch,shuffle=True)      #读训练数据,shuffle=True表示每次迭代都打乱数据
        # 将113315个（x1,y1,label)按照32大小分割，每一个batch格式（32,x1,x2,label）,(x1,x2)是cd索引,对应一对circRNA与disease,label是cd在(x1,x2)的数值
        testSet=DataLoader(MyDataset(testSet_index[i],cd),batch,shuffle=False)     #读测试数据
        #（网络，损失函数，优化器，训练数据，测试数据，训练集特征矩阵，伦次，折号）
        train(net, cost, optimizer, trainSet, testSet, features[i], epoch, i)    #训练
        
         
    mean_fprs, mean_tprs, mean_recalls, mean_precs= [], [], [], []
    # cd, (834, 138); tei, [fold1 测试集索引, fold2 测试集索引, fold3 测试集索引, fold4 测试集索引, fold5 测试集索引]
    _,_,_,tei=torch.load('./circ_drug_CNN_1_1_util.pth')
    # 以疾病为主体，
    for i in range(5):
        pred, y=torch.load('./circ_CNNplt_%d'%i)
        # (2, 113315)>> (113315, 2), 第一列是cRNA, 第二列是疾病
        test_idx= tei[i].T
        # dim trans
        test_idx= torch.stack([test_idx[:, 1], test_idx[:, 0]], dim= 1) # (113315, 2)
        mean_fpr, mean_tpr, mean_recall, mean_prec= plt.roc_pr4_folder(test_idx, y, pred[:, 1], (218, 271))
        mean_fprs.append(torch.tensor(mean_fpr)); mean_tprs.append(torch.tensor(mean_tpr)); mean_recalls.append(torch.tensor(mean_recall)); mean_precs.append(torch.tensor(mean_prec))
    mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts= torch.stack(mean_fprs), torch.stack(mean_tprs), torch.stack(mean_recalls, dim= 0), torch.stack(mean_precs, dim= 0)
    plt.roc_pr4cross_val(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, 5)
    #移动到指定目录,如果不存在则创建
    if not os.path.exists('./test%d'%times):
        os.makedirs('./test%d'%times)
    # 移动文件
    os.rename('./roc_curve.png', './test%d/roc_curve.png'%times)
    os.rename('./pr_curve.png', './test%d/pr_curve.png'%times)


print('-----------------------------------------------------------------------finish-----------------------------------------------------------------------')


'''===========一次性计算==========='''
# # 计算miRNA-miRNA相似度矩阵，算一次就行了，保存到文件
# mm = data_loader.calculate_sim(dm.T, dd)
# print(mm[:5, :5])
# # 把cc保存到文件,格式为npy
# np.save("mycode\data\miRNA_miRNA.npy", mm)
'''===========一次性计算==========='''


'''===========一次性计算==========='''
# # 加载数据 circRNA-disease, disease-disease, disease-miRNA, circRNA-miRNA
# cd, dd, dm, cm, mm= data_loader.load_data()
# # 5折，训练集正负比例1：1
# k = 5
# neg_ratio = 1
# #得到训练集   测试集   遮盖后测试集正例后的关联矩阵  circ_circ_sim
# trainSet, testSet, cda, cc = data_loader.split_dataset(cd, dd, k, neg_ratio)

# feas = []

# #生成5折的邻接矩阵，也是特征矩阵！！！！
# for i in range(k):
#     fea = data_loader.cfm(cc[i], cda[i], dd, dm, cm, mm) # 特征矩阵的cd已经针对测试集进行了覆盖
#     feas.append(fea)

# # cd(未遮盖的！！)   邻接/特征矩阵  训练集  测 集
# torch.save([cd, feas, trainSet, testSet], 'circ_CNN.pth')
'''===========一次性计算==========='''