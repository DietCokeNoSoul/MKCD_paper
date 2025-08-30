import numpy as np
import torch


'''
    k折交叉验证
    assMatrix -- circRNA-disease关联矩阵,即带标签矩阵
    k  ---  cross数
    negr --- 1 每个正样本配备的负样本数,设置1 表示正负样本1:1(训练集)
    
    rand_index --- 随机索引(0-所有正例数)
    pos_index --- 正样本索引
    neg_index --- 负样本索引
'''
def load_data():  # circ 834  dis 138  mi 555
    cd = np.load("mycode\data\circ_disease\circRNA_disease.npy")  # 834*138
    # dis_dis_sim也直接给不用计算
    dd = np.load("mycode\data\circ_disease\disease_disease.npy")  # 138*138
    dm = np.load("mycode\data\circ_disease\disease_miRNA.npy")  # 138*555
    cm = np.load("mycode\data\circ_disease\circRNA_miRNA.npy")  # 834*555
    mm = np.load("mycode\data\circ_disease\miRNA_miRNA.npy")  # 555*555
    return torch.tensor(cd), torch.tensor(dd), torch.tensor(dm), torch.tensor(cm), torch.tensor(mm)  # 返回tensor格式的数据

def split_dataset(assMatrix, po_index):  #5 cross  5折
    #随机生成索引,cd的数值类型为double，sum结果也为double，long转换为整数tensor,item转换为int,randperm生成0到正例总数的随机排列,0~989
    cda = assMatrix.clone()  # 复制一份assMatrix
    cda[po_index[0], po_index[1]] = 0  # 将正例的值置为0，遮盖正例
    rand_index = torch.randperm(cda.sum().long().item()) # 989*1
    # argwhere返回非零元素的索引，index_select根据索引取值，# index_select根据索引取值,0表示按行取值,rand_index表示随机索引表，将按照rand_index的顺序取值，最后组成一个新的索引表
    pos_index = torch.argwhere(cda == 1).index_select(0, rand_index).T #2*989,随机过后的索引，每一列都是一个正例索引，即cd中有关联的rna疾病对的位置
    neg_index = torch.argwhere(assMatrix == 0) # 114103*2,每一行都是一个负例索引，即cd中没有关联的rna疾病对的位置
    neg_index = neg_index.index_select(0, torch.randperm(neg_index.shape[0])).T  #2*114103,随机过后的索引，每一列都是一个负例索引，即cd中没有关联的rna疾病对的位置
    #循环生成每折数据,分成k份，k-1份正样本+同等数量的负样本=训练集，1份正样本+去除k份参与过训练的其余负样本=测试集
    '''分割正负样本'''
    # 分割正样本：第i折为测试集，其他为训练集
    pos_Sample = pos_index #正样本组成训练集,pos_Sample:2*922,一列对应一个正例索引的行列号
    # 取前922个负样本
    neg_Sample = neg_index[:, :pos_Sample.shape[1]]
    
    #正负样本组成训练集
    trainData = torch.cat([pos_Sample, neg_Sample], dim=1) #trainData:2*1576,一列对应一个正例索引的行列号
    #第i折正负样本作为测试集
    # test_neg_count = 70
    test_neg_samples = neg_index[:, pos_Sample.shape[1]:]
    testData = torch.cat([po_index, test_neg_samples], dim=1) # testData: 2*(正例数+70)
    #测试集
    return trainData, testData, cda  #测试集大小为5,每个元素为2*n

disease_name = np.load("mycode\data\circ_disease\disease_name.npy")
cd, dd, dm, cm, mm= load_data()

# 找出每个疾病关联的circRNA数量
disease_counts = cd.sum(dim=0)  # 按列求和，得到每个疾病的关联数

# 挑出关联最多的两个疾病的索引
top2_indices = torch.topk(disease_counts, 2).indices.tolist()

# 输出疾病名称
for idx in top2_indices:
    print(f"疾病名称: {disease_name[idx]} 关联circRNA数量: {disease_counts[idx].item()}")

# 每个疾病的所有关联的行列号
top2_tensors = []
for idx in top2_indices:
    circ_indices = torch.nonzero(cd[:, idx]).squeeze()
    rows = circ_indices if circ_indices.ndim == 1 else circ_indices.flatten()
    cols = torch.full((rows.shape[0],), idx, dtype=torch.long)
    tensor = torch.stack([rows, cols], dim=0)
    top2_tensors.append(tensor)

for i, tensor in enumerate(top2_tensors):
    print(f"第{i+1}个疾病的关联行列号张量:\n{tensor}")

# 额外拿出glioma疾病的关联
glioma_idx = None
for i, name in enumerate(disease_name):
    if str(name).lower().find('glioma') != -1:
        glioma_idx = i
        break

if glioma_idx is not None:
    circ_indices = torch.nonzero(cd[:, glioma_idx]).squeeze()
    rows = circ_indices if circ_indices.ndim == 1 else circ_indices.flatten()
    cols = torch.full((rows.shape[0],), glioma_idx, dtype=torch.long)
    glioma_tensor = torch.stack([rows, cols], dim=0)
    print(f"glioma疾病的关联行列号张量:\n{glioma_tensor}")
    print(f"glioma疾病的关联对数量: {glioma_tensor.shape[1]}")
else:
    print("未找到glioma疾病。")
    

disease_test_dataset_index_1 = top2_tensors[0]
disease_test_dataset_index_2 = top2_tensors[1]
disease_test_dataset_index_glioma = glioma_tensor

single_trainset_1, single_testset_1, cd_1 = split_dataset(cd, disease_test_dataset_index_1)
single_trainset_2,single_testset_2, cd_2 = split_dataset(cd, disease_test_dataset_index_2)
single_trainset_glioma,single_testset_glioma, cd_3 = split_dataset(cd, disease_test_dataset_index_glioma)

torch.save([single_trainset_1, single_trainset_2, single_trainset_glioma], 'single_disease_trainsets.pth')
torch.save([single_testset_1, single_testset_2, single_testset_glioma], 'single_disease_testsets.pth')
torch.save([cd_1, cd_2, cd_3], 'single_disease_cds.pth')