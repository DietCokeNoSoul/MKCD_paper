'''
任务:采用CNN来对circRNA-disease数据进行预测,并通过无责交叉验证,然后输出AUC和AUPR
数据集:circRNA_disease、circRNA_miRNA、disease_disease、disease_miRNA ||| circRNA_name、disease_name、miRNA_name
'''

# 导入第三方包
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
import os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

# 导入自定义包
import CNN_Model
from data_loader import MyDataset
from util import plt
from train_function import train
from RWR.rwr import combine_rwr
from RWR.rwr import bfs_topology_embedding
from RWR.rwr import pagerank_topology_embedding
import random
import numpy as np

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
'''
    数据加载与预处理
'''
# 读取数据 circRNA-disease5*834*138 覆盖过得特征矩阵5*1527*1527 训练集索引表5*2*1576 测试集索引表5*2*113315
_, cd, features, trainSet_index, testSet_index = torch.load('circ_CNN.pth') 
# cd, features, trainSet_index, testSet_index = torch.load('circ_CNN_caseStudy.pth') 
print(cd.shape, features[0].shape, trainSet_index[0].shape, testSet_index[0].shape)
# trainSet_index = torch.load('single_disease_trainsets.pth')[0]
testSet_index[0] = torch.load('single_disease_testsets.pth')[0]
# cd = torch.load('single_disease_cds.pth')[0]
# print(trainSet_index.shape, testSet_index.shape, cd.shape)
# # 从features[0]中取出相似矩阵cc，cc为834*834的矩阵，表示circRNA与circRNA之间的关系
# cc = features[0][:834, :834]
# # 设置对角线为0
# cc = cc - torch.diag(torch.diag(cc))
# avg = cc.sum() / (834 * 834) # 0.0361
# # 取出相似矩阵dd，dd为138*138的矩阵，表示disease与disease之间的关系
# dd = features[0][834:972, 834:972]
# dd = dd - torch.diag(torch.diag(dd))
# avg = dd.sum() / (138 * 138) # 0.0188
# # 取出相似矩阵mm，mm为555*555的矩阵，表示miRNA与miRNA之间的关系
# mm = features[0][972:, 972:]
# mm = mm - torch.diag(torch.diag(mm))
# avg = mm.sum() / (555 * 555) # 0.0959

'''
    超参数设置
'''


learn_rate=1e-3   #学习率
epoch=10            #伦次
batch=32            #批大小
#2058, 2048, 2049, 2050, 2057
seedIndex = [1027, 2049, 1024, 105, 95] 
seed = [20, 10, 20, 15, 12]
rwr_index = [0.3, 0.5, 0.2]
'''
    训练模型
'''       


mean_fprs, mean_tprs, mean_recalls, mean_precs= [], [], [], []
mean_auc, mean_aupr = [],[]
# for times in range(len(seed)):
# for times in range(5):
for i in range(1): #5折交叉验证
# for i in cross: #5折交叉验证
    # if i == 0:
    #     continue
    # if i == 1:
    #     continue
    # if i == 2:
    #     continue
    # if i == 3:
    #     continue
    # if i == 4:
    #     continue
    # set_seed(seed=seedIndex[i])
    # print("----------------------seed----------------------", seedIndex[i])
    print('cross:%d'%(i+1))
    net=CNN_Model.Net().to(device)  #放入cuda
    cost=nn.CrossEntropyLoss()      #交叉熵损失
    #优化器  Adam   参数（模型参5数，学习率，l2正则化）
    optimizer=torch.optim.AdamW(net.parameters(),learn_rate,weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    # 将1576个（x1,y1,label)按照32大小分割，每一个batch格式（32,x1,x2,label）,(x1,x2)是cd索引,对应一对circRNA与disease,label是cd在(x1,x2)的数值
    trainSet=DataLoader(MyDataset(trainSet_index[i],cd),batch,shuffle=True)      #读训练数据,shuffle=True表示每次迭代都打乱数据
    # 将113315个（x1,y1,label)按照32大小分割，每一个batch格式（32,x1,x2,label）,(x1,x2)是cd索引,对应一对circRNA与disease,label是cd在(x1,x2)的数值
    testSet=DataLoader(MyDataset(testSet_index[i],cd),batch,shuffle=False)     #读测试数据
    # 随机游走
    rwr = combine_rwr(rwr_index, features[i])
    # rwr = bfs_topology_embedding(features[i], depth=2).to(device)
    para=[]
    # for k in range(41):
    #     para.append([])
    # para.append([1, 1])
    #（网络，损失函数，优化器，训练数据，测试数据，训练集特征矩阵，伦次，折号）
    train(net,cost,optimizer,trainSet,testSet,features[i],epoch,i, scheduler, rwr,device,para)    #训练
    # print(para)
    # torch.save(para, './para')  #保存模型参数

for i in range(1):
    pred, y = torch.load('./circ_CNNplt_%d' % i, map_location=torch.device(device))
    pred = torch.softmax(pred, dim=1)[:, 1]
    # pred, y = torch.load('./circ_CNNplt_4', map_location=torch.device(device))
    # (2, 113315)>> (113315, 2), 第一列是cRNA, 第二列是疾病
    test_idx = testSet_index[i].T
    # dim trans
    test_idx = torch.stack([test_idx[:, 1], test_idx[:, 0]], dim=1)  # (113315, 2)
    mean_fpr, mean_tpr, mean_recall, mean_prec, aucN, auprN = plt.roc_pr4_folder(test_idx, y, pred, (138, 834))
    mean_auc.append(aucN)
    mean_aupr.append(auprN)
    # 打印auc和aupr的平均值
    print("Mean AUC: ", torch.mean(torch.tensor(mean_auc)))
    print("Mean AUPR: ", torch.mean(torch.tensor(mean_aupr)))
    print()
    mean_fprs.append(torch.tensor(mean_fpr)); mean_tprs.append(torch.tensor(mean_tpr)); mean_recalls.append(torch.tensor(mean_recall)); mean_precs.append(torch.tensor(mean_prec))
# mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts= torch.stack(mean_fprs), torch.stack(mean_tprs), torch.stack(mean_recalls, dim= 0), torch.stack(mean_precs, dim= 0)
# torch.save([mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts],'My.pkl')
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