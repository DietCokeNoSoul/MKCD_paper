import numpy as np
import itertools
import torch
from torch.utils.data import Dataset


'''
    读取数据
    返回值:cd, dd, dm, cm, mm,分别是circRNA-disease, disease-disease, disease-miRNA, circRNA-miRNA, miRNA-miRNA
'''
def load_data():  # circ 834  dis 138  mi 555
    cd = np.load("mycode\data\circ_disease\circRNA_disease.npy")  # 834*138
    # dis_dis_sim也直接给不用计算
    dd = np.load("mycode\data\circ_disease\disease_disease.npy")  # 138*138
    dm = np.load("mycode\data\circ_disease\disease_miRNA.npy")  # 138*555
    cm = np.load("mycode\data\circ_disease\circRNA_miRNA.npy")  # 834*555
    mm = np.load("mycode\data\circ_disease\miRNA_miRNA.npy")  # 555*555
    return torch.tensor(cd), torch.tensor(dd), torch.tensor(dm), torch.tensor(cm), torch.tensor(mm)  # 返回tensor格式的数据


'''
    计算circRNA-circRNA相似度矩阵
    cd -- circRNA-disease相似度矩阵
    dd -- disease-disease相似度矩阵
'''
def calculate_sim(cd, dd):  # cd:834*138 dd:138*138
    s1 = cd.shape[0]  # 834 circRNA数
    cc = torch.eye(s1)  # 生成一个对角线为1的矩阵,大小为834*834
    m2 = dd * cd[:,None, :]  # 138*138*834，dd为138*138，cd为834*138，cd[:, None, :]为834*1*138
    m1 = cd[:, :, None]  # 834*138*1
    for x, y in itertools.permutations(torch.linspace(0, s1 - 1, s1, dtype=torch.long),2):  # 生成排列组合，x为0-833，y为0-833，不包括相同的，如(0,0)，(1,1)等。
        x, y = x.item(), y.item()  # 将tensor转换为int
        m = m1[x, :, :] * m2[y, :, :]  # 834*138*138，m1[x, :, :]为1*138*1，m2[y, :, :]为138*138*834
        if cd[x].sum() + cd[y].sum() == 0:  # 如果cd[x]和cd[y]的和为0
            cc[x, y] = 0  # 则cc[x, y]为0
        else:
            cc[x, y] = (m.max(dim=0, keepdim=True)[0].sum() +m.max(dim=1, keepdim=True)[0].sum()) / (cd[x].sum() + cd[y].sum())  # 否则cc[x, y]为m中最大值的和除以cd[x]和cd[y]的和，即相似度
    return cc  # 返回相似度矩阵


'''
    k折交叉验证
    assMatrix -- circRNA-disease关联矩阵,即带标签矩阵
    dd -- disease-disease相似度矩阵
    k  ---  cross数
    negr --- 1 每个正样本配备的负样本数,设置1 表示正负样本1:1(训练集)
    
    rand_index --- 随机索引(0-所有正例数)
    pos_index --- 正样本索引
    neg_index --- 负样本索引
'''
def split_dataset(assMatrix, dd, k, negr):  #5 cross  5折
    #训练集索引表   测试集索引表   遮盖后测试集正例后的关联矩阵  circ_circ_sim
    trainSet_index, testSet_index, cd_mask, cc = [], [], [], []
    #随机生成索引,cd的数值类型为double，sum结果也为double，long转换为整数tensor,item转换为int,randperm生成0到正例总数的随机排列,0~989
    rand_index = torch.randperm(assMatrix.sum().long().item()) # 989*1
    # argwhere返回非零元素的索引，index_select根据索引取值，# index_select根据索引取值,0表示按行取值,rand_index表示随机索引表，将按照rand_index的顺序取值，最后组成一个新的索引表
    pos_index = torch.argwhere(assMatrix == 1).index_select(0, rand_index).T #2*989,随机过后的索引，每一列都是一个正例索引，即cd中有关联的rna疾病对的位置
    neg_index = torch.argwhere(assMatrix == 0) # 114103*2,每一行都是一个负例索引，即cd中没有关联的rna疾病对的位置
    neg_index = neg_index.index_select(0, torch.randperm(neg_index.shape[0])).T  #2*114103,随机过后的索引，每一列都是一个负例索引，即cd中没有关联的rna疾病对的位置
    crossCount = int(pos_index.shape[1] / k) # 每折的样本数为989/5=197
    #循环生成每折数据,分成k份，k-1份正样本+同等数量的负样本=训练集，1份正样本+去除k份参与过训练的其余负样本=测试集
    for i in range(k):
        '''分割正负样本'''
        # 分割正样本：第i折为测试集，其他为训练集
        pos_Sample = torch.cat([pos_index[:, :(i * crossCount)], pos_index[:, ((i + 1) * crossCount):(k * crossCount)]],dim=1) #正样本组成训练集,pos_Sample:2*788,一列对应一个正例索引的行列号
        neg_Sample = torch.cat([neg_index[:, :(i * crossCount * negr)], neg_index[:,((i + 1) * crossCount * negr):(k * crossCount * negr)]],dim=1) #负样本组成训练集,neg_Sample:2*788,一列对应一个负例索引的行列号
        #正负样本组成训练集
        trainData = torch.cat([pos_Sample, neg_Sample], dim=1) #trainData:2*1576,一列对应一个正例索引的行列号
        #第i折正负样本作为测试集
        testData = torch.cat([pos_index[:, (i * crossCount):((i + 1) * crossCount)], neg_index[:, (k * crossCount * negr):]], dim=1) #testData:2*113315,一列对应一个负例索引的行列号
        #当前训练集 测试集 分别存储
        trainSet_index.append(trainData) #trainSet_index大小为5,每个元素为2*1576
        testSet_index.append(testData) #testSet_index大小为5,每个元素为2*113315
        '''计算每折的circRNA-circRNA相似度矩阵'''
        # 克隆一个关联矩阵并将当前折的正样本置零（去掉当前测试集中的正样本）  遮盖测试集正例！！！！！
        cdt = assMatrix.clone()
        cdt[pos_index[0, (i * crossCount):((i + 1) * crossCount)], pos_index[1, (i * crossCount):((i + 1) * crossCount)]] = 0 # 把当前测试集的正例遮盖掉
        #保存遮盖后的circ_dis_asso
        cd_mask.append(cdt)# 共有5个cdt,每个cdt大小为834*138，分别都对不同测试集的正例进行了遮盖
        #计算circ_circ_association
        cc.append(calculate_sim(cdt, dd))
    # 返回训练集   测试集   遮盖后测试集正例后的关联矩阵  circ_circ_sim
    return trainSet_index, testSet_index, cd_mask, cc #训练集大小为5,每个元素为2*1576,测试集大小为5,每个元素为2*113315,遮盖后的关联矩阵大小为5,每个元素为834*138,circRNA-circRNA相似度矩阵大小为5,每个元素为834*834
  

'''
    生成特征矩阵
    cc -- circRNA-circRNA相似度矩阵
    cd -- circRNA-disease相似度矩阵
    dd -- disease-disease相似度矩阵
    dm -- disease-miRNA相似度矩阵
    cm -- circRNA-miRNA相似度矩阵
    mm -- miRNA-miRNA相似度矩阵
    feature -- 特征矩阵
'''
def cfm(cc, cd, dd, dm, cm, mm): # cc:834*834 cd:834*138 dd:138*138 md:138*555 cm:834*555 mm:555*555
    r1 = torch.cat([cc, cd, cm], dim=1)  #按列拼接,r1:834*1527，行为circRNA,列为circRNA, disease, miRNA
    r2 = torch.cat([cd.T, dd, dm], dim=1) #按列拼接,r2:138*1527，行为disease,列为circRNA, disease, miRNA
    r3 = torch.cat([cm.T, dm.T, mm], dim=1) #按列拼接,r3:555*1527，行为miRNA,列为circRNA, disease, miRNA
    feature = torch.cat([r1, r2, r3], dim=0) #按行拼接,fea:1527*1527，行为circRNA, disease, miRNA,列为circRNA, disease, miRNA
    return feature


'''
    获取数据集
    输入为cd和索引表
    输出为cd中的数据
'''
#用于处理自己的数据，输出坐标和cd矩阵，返回坐标和标签
class MyDataset(Dataset):
    def __init__(self,tri,cd):
        self.tri=tri
        self.cd=cd
    def __getitem__(self,idx):
        x,y=self.tri[:,idx]
        label=self.cd[x][y]
        return x,y,label
    def __len__(self):
        return self.tri.shape[1]
   
    
# # 加载数据 circRNA-disease, disease-disease, disease-miRNA, circRNA-miRNA
# cd, dd, dm, cm, mm= load_data()
# # 5折，训练集正负比例1：1
# k = 5
# neg_ratio = 1
# #得到训练集   测试集   遮盖后测试集正例后的关联矩阵  circ_circ_sim
# trainSet, testSet, cda, cc = split_dataset(cd, dd, k, neg_ratio)

# feas = []

# #生成5折的邻接矩阵，也是特征矩阵！！！！
# for i in range(k):
#     fea = cfm(cc[i], cd, dd, dm, cm, mm) # 特征矩阵的cd已经针对测试集进行了覆盖
#     feas.append(fea)

# # cd(未遮盖的！！)   邻接/特征矩阵  训练集  测 集
# torch.save([cd, feas, trainSet, testSet], 'circ_CNN_caseStudy.pth')