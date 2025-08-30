import numpy as np
import pandas as pd
import itertools
import torch
from torch.utils.data import Dataset
import GIP
import snf

import os
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)
from util import util


'''
    读取数据
    返回值:cd, dd, dm, cm, mm,分别是circRNA-disease, disease-disease, disease-miRNA, circRNA-miRNA, miRNA-miRNA
'''
def load_data():  # circ 834  dis 138  mi 555
    cd = pd.read_csv('mycode\data\circrna_drug_data\\association.csv')
    cd = torch.tensor(cd.values)  # 转换为tensor
    cd = cd[:,1:]# 去除第一列
    
    ccs = pd.read_csv('mycode\data\circrna_drug_data\\gene_seq_sim.csv')
    ccs = torch.tensor(ccs.values)
    ccs = ccs[:,1:]
    
    dds = pd.read_csv('mycode\data\circrna_drug_data\\drug_str_sim.csv')
    dds = torch.tensor(dds.values)
    dds = dds[:,1:]
    return cd, ccs, dds  # 返回tensor格式的数据


'''
    k折交叉验证
    assMatrix -- circRNA-disease关联矩阵,即带标签矩阵
    k  ---  cross数
    negr --- 1 每个正样本配备的负样本数,设置1 表示正负样本1:1(训练集)
    
    rand_index --- 随机索引(0-所有正例数)
    pos_index --- 正样本索引
    neg_index --- 负样本索引
'''
def split_dataset(assMatrix, ccs, dds, k, negr):  #5 cross  5折
    #训练集索引表   测试集索引表   遮盖后测试集正例后的关联矩阵  circ_GIP_sim  drug_GIP_sim
    trainSet_index, testSet_index, cd_mask, cc, dd = [], [], [], [], []
    # randperm随机生成索引,cd的数值类型为double，sum结果也为double，long转换为整数tensor,item转换为int,randperm生成0到正例总数的随机排列,0~989
    # argwhere返回非零元素的索引，index_select根据索引取值，# index_select根据索引取值,0表示按行取值,rand_index表示随机索引表，将按照rand_index的顺序取值，最后组成一个新的索引表
    pos_index = torch.argwhere(assMatrix == 1) # 2*4134,每一列都是一个正例索引，即cd中有关联的rna疾病对的位置
    pos_index = pos_index.index_select(0, torch.randperm(pos_index.shape[0])).T #2*4134,随机过后的索引，每一列都是一个正例索引，即cd中有关联的rna疾病对的位置
    neg_index = torch.argwhere(assMatrix == 0) # 54944*2,每一行都是一个负例索引，即cd中没有关联的rna疾病对的位置
    neg_index = neg_index.index_select(0, torch.randperm(neg_index.shape[0])).T  #2*54944,随机过后的索引，每一列都是一个负例索引，即cd中没有关联的rna疾病对的位置
    crossCount = int(pos_index.shape[1] / k) # 每折的样本数为4134/5=826
    #循环生成每折数据,分成k份，k-1份正样本+同等数量的负样本=训练集，1份正样本+去除k份参与过训练的其余负样本=测试集
    for i in range(k):
        '''分割正负样本'''
        # 分割正样本：第i折为测试集，其他为训练集
        pos_Sample = torch.cat([pos_index[:, :(i * crossCount)], pos_index[:, ((i + 1) * crossCount):(k * crossCount)]],dim=1) #正样本组成训练集,pos_Sample:2*3304,一列对应一个正例索引的行列号
        neg_Sample = torch.cat([neg_index[:, :(i * crossCount * negr)], neg_index[:,((i + 1) * crossCount * negr):(k * crossCount * negr)]],dim=1) #负样本组成训练集,neg_Sample:2*3304,一列对应一个负例索引的行列号
        #正负样本组成训练集
        trainData = torch.cat([pos_Sample, neg_Sample],dim=1) #trainData:2*6608,一列对应一个正例索引的行列号
        #第i折正负样本作为测试集
        testData = torch.cat([pos_index[:, (i * crossCount):((i + 1) * crossCount)], neg_index[:, ((k + 1 + i) * crossCount * negr):((k + 2 + i) * crossCount * negr)]],dim=1)#testData:2*51640,一列对应一个负例索引的行列号
        #当前训练集 测试集 分别存储
        trainSet_index.append(trainData) #trainSet_index大小为5,每个元素为2*6608
        testSet_index.append(testData) #testSet_index大小为5,每个元素为2*51640
        '''计算每折的circRNA-circRNA相似度矩阵'''
        # 克隆一个关联矩阵并将当前折的正样本置零（去掉当前测试集中的正样本）  遮盖测试集正例！！！！！
        cdt = assMatrix.clone()
        cdt[pos_index[0, (i * crossCount):((i + 1) * crossCount)], pos_index[1, (i * crossCount):((i + 1) * crossCount)]] = 0 # 把当前测试集的正例遮盖掉
        #保存遮盖后的circ_dis_asso
        cd_mask.append(cdt)# 共有5个cdt,每个cdt大小为834*138，分别都对不同测试集的正例进行了遮盖
        #计算circ_GIP_sim与drug_GIP_sim
        # ccg = GIP.compute_gip_kernel(cdt)
        # ddg = GIP.compute_gip_kernel(cdt.T)
        # snf
        # cc.append(torch.tensor(snf.snf(ccs, ccg, K=20, t=20, alpha=1.0)))
        # dd.append(torch.tensor(snf.snf(dds, ddg, K=20, t=20, alpha=1.0)))
        ccf, ddf = util.get_syn_sim(cdt, ccs, dds, 1)
        cc.append(torch.tensor(ccf))
        dd.append(torch.tensor(ddf))
    # 返回训练集   测试集   遮盖后测试集正例后的关联矩阵  circ_GIP_sim  drug_GIP_sim
    return trainSet_index, testSet_index, cd_mask, cc, dd #训练集大小为5,每个元素为2*6608,测试集大小为5,每个元素为2*51640,遮盖后的关联矩阵大小为5,每个元素为271*218,circRNA-circRNA相似度矩阵大小为5,每个元素为271*271,drug-drug相似度矩阵大小为5,每个元素为218*218


'''
    生成特征矩阵
    cc -- circRNA-circRNA相似度矩阵
    cd -- circRNA-drug关联矩阵
    dd -- drug-drug相似度矩阵
    feature -- 特征矩阵
'''
def cfm(cc, cd, dd): # cc:271*271 cd:271*218 dd:218*218
    r1 = torch.cat([cc, cd], dim=1)  #按列拼接,r1:271*489，行为circRNA,列为circRNA, drug
    r2 = torch.cat([cd.T, dd], dim=1) #按列拼接,r2:218*489，行为drug,列为circRNA, drug
    feature = torch.cat([r1, r2], dim=0) #按行拼接,fea:489*489，行为circRNA, drug,列为circRNA, drug
    return feature


'''
    获取数据集
    输入为cd和索引表
    输出为cd中的数据
'''
class MyDataset(Dataset):
    def __init__(self,dataSet,cd):
        self.dataSet=dataSet
        self.cd=cd
    def __getitem__(self,idx):
        x,y=self.dataSet[:,idx]
        label=self.cd[x][y]
        return x,y,label
    def __len__(self):
        return self.dataSet.shape[1]
    
    

'''===========一次性计算==========='''
cd, ccs, dds = load_data()

# 5折，训练集正负比例1：1
k = 5
neg_ratio = 1
#得到训练集   测试集   遮盖后测试集正例后的关联矩阵  circ_snf_sim  drug_snf_sim
trainSet, testSet, cda, cc, dd = split_dataset(cd, ccs, dds, k, neg_ratio)

feas = []
#生成5折的邻接矩阵，也是特征矩阵！！！！
for i in range(k):
    fea = cfm(cc[i], cda[i], dd[i]) # 特征矩阵的cd已经针对测试集进行了覆盖
    feas.append(fea)    

# cd(未遮盖的！！)   邻接/特征矩阵  训练集  测 集
torch.save([cd, feas, trainSet, testSet], 'circ_drug_CNN_1_1_util.pth')
'''===========一次性计算==========='''
# cd, features, trainSet_index, testSet_index = torch.load('DPMGCDA_1_1.pth')
# for i in range(k):
#     print(cd.shape, features[i].shape, testSet_index[i][1].shape,testSet_index[i][0].shape)


