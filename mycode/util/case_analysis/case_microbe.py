from scipy.sparse import coo_matrix
import numpy as np
import torch
import os


# import numpy as np
# top candidate 列举
# def topk_candidate_microbles2file(topk=20, epoch=79):
#     # 读取所有微生物的名字
#     microbe_names = []
#     with open('../data/microbe_names.txt', 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             microbe_names.append(line.rstrip())
#     microbe_names = np.array(microbe_names)
#     # 读取所有药物的名字
#     drug_names = []
#     with open('../data/drug_names.txt', 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             drug_names.append(line.rstrip().strip('\ufeff'))
#     drug_names = np.array(drug_names)
#     # print(drug_names)
#     # 读取预测的结果
#     microbe_result = []
#     microbe_result = np.loadtxt(f'./run_epoch/{epoch}/case_analysis_result_all_drugs.txt')
#     microbe_idx = np.argsort(-microbe_result, axis=1)[:, 0: topk]
#     # 获取每个药物对应的前20个微生物的名字与预测值
#     ls = []
#     for i in range(len(drug_names)):
#         di_names = np.array([drug_names[i]] * topk).reshape(-1, 1)
#         di_microbe_names = microbe_names[microbe_idx[i]].reshape(-1, 1)
#         rank = np.arange(1, topk + 1, 1).reshape(-1, 1)
#         probability = microbe_result[i][microbe_idx[i]].reshape(-1, 1)
#         ls.append(np.concatenate((di_names, rank, di_microbe_names, probability), axis=1))
#     # idx= np.array([598, 970, 1343])
#     drug_i_idx = 1343
#     print(ls[drug_i_idx])
#     # 保存文件
#     result932690 = np.concatenate(ls, axis=0)
#     # print(result950807.shape)
#     np.savetxt(f'./run_epoch/{epoch}/candidate_microbes.txt', result932690, fmt='%s', delimiter='\t', encoding='utf-8')
#
#
# # top candidate 列举
# def topk_candidate_microbles(epoch, topk=20):
#     # 读取所有微生物的名字
#     microbe_names = []
#     with open('../compared_method/microbe_names.txt', 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             microbe_names.append(line.rstrip())
#     # 读取预测的结果
#     microbe_result = []
#     microbe_result = torch.from_numpy(np.loadtxt(f'./epoch/{epoch}epoch/case_analysis_result_{epoch}.txt'))
#     microbe_idx = microbe_result.sort(dim=1, descending=True)[1][:, 0: topk].to(torch.long)
#     # 输出top
#     # for i in range(topk):
#     # print(f'CIPROFLOXACIN top {i+ 1} candidate microbe: {microbe_names[microbe_idx[0, i]]}; Moxifloxacin top {i+ 1} candidate microbe: {microbe_names[microbe_idx[1, i]]};')
#     for i in range(topk):
#         print(
#             f'CIPROFLOXACIN top {i + 1} candidate microbe: {microbe_names[microbe_idx[0, i]]}; Moxifloxacin top {i + 1} candidate microbe: {microbe_names[microbe_idx[1, i]]}; Vancomycin top {i + 1} candidate microbe: {microbe_names[microbe_idx[2, i]]};')


# ciprofloxacin与Moxifloxacin关联微生物列举
def ls_candidate_microbles_name(database, drug_name, dir1='../data'):
    # 提取相应行
    drug_names, microbe_names = [], []
    with open(os.path.join(dir1, database, 'drugs.txt'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            drug_names.append(line.rstrip())
    with open(os.path.join(dir1, database, 'microbes.txt'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            microbe_names.append(line.rstrip())
    drug_idx = drug_names.index(drug_name)

    # 重构关联矩阵
    adj_info = np.loadtxt(os.path.join(dir1, database, 'adj.txt'))
    col1, col2, col3 = adj_info[:, 0], adj_info[:, 1], adj_info[:, 2]
    adj_mat = coo_matrix((col3, (col1 - 1, col2 - 1)), shape=(len(drug_names), len(microbe_names)),
                         dtype=int).toarray()

    # 返回对应候选名
    microbe_candidate_idxs = np.where(adj_mat[drug_idx, :] == 1)[0]
    # print(microbe_candidate_idxs)
    print(f'in {database}, {drug_name} association: ')
    for idx in microbe_candidate_idxs:
        print(microbe_names[idx])
    print('\n')


# 哪些药物有很多关联的微生物
def ls_big_drug(database, dir1='../data'):
    # 提取相应行
    drug_names, microbe_names = [], []
    with open(os.path.join(dir1, database, 'drugs.txt'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            drug_names.append(line.rstrip())
    with open(os.path.join(dir1, database, 'microbes.txt'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            microbe_names.append(line.rstrip())

    # 重构关联矩阵
    adj_info = np.loadtxt(os.path.join(dir1, database, 'adj.txt'))
    col1, col2, col3 = adj_info[:, 0], adj_info[:, 1], adj_info[:, 2]
    adj_mat = coo_matrix((col3, (col1 - 1, col2 - 1)), shape=(len(drug_names), len(microbe_names)),
                         dtype=int).toarray()
    adj_mat = torch.from_numpy(adj_mat)
    #
    drug_association_number = adj_mat.sum(dim=1)
    torch.set_printoptions(profile="full")
    a = torch.sort(drug_association_number, descending=True)
    # 前10个关联数
    ass_num = a[0][0:10].numpy()
    j = 0
    # 前10个关联最多的药物的索引
    name_index = a[1][0:10].numpy()
    for i in name_index:
        # 输出前10个关联最多的药物的索引 名字 关联数
        print(i, drug_names[i], ass_num[j])
        j += 1
    # print((torch.sort(drug_association_number, descending=True)[0])[0: 10])
    # print((torch.sort(drug_association_number, descending=True)[1])[0: 10])


# Moxifloxacin
# ls_big_drug('MDAD')
# 633 Curcumin 18
# 731 Epigallocatechin Gallate 16
# 598 Ciprofloxacin 10
# 1343 Vancomycin 10
# 922 LL-37 9
# 752 Farnesol 8
# 850 Indole 8
# 813 Hamamelitannin 8
# 1313 Tobramycin 8
# 569 Ceftazidime 8
# 970 Moxifloxacin

# ls_candidate_microbles_name('mdad', 'Vancomycin')
# ls_candidate_microbles_name('aBiofilm', 'Ciprofloxacin')
ls_candidate_microbles_name('aBiofilm', 'Moxifloxacin')
ls_candidate_microbles_name('mdad', 'Moxifloxacin')
