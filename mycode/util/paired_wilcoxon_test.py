from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
from scipy.stats import wilcoxon as w_test
import numpy as np
import torch
import os


# 根据五折结果，算一折每个实体的auc与aupr，再对折数取平均
def compute_methodA_avg_auc_aupr4every_entity(ass_mat_shape, fold_num=5, fold5_dir='mycode/CNN_disease/TestData/Wilcoxon/xxx/', save_dir='mycode/CNN_disease/TestData/Wilcoxon/xxx/'):
    # input, 关联矩阵大小, e.g. (1373, 173); fold_num, 哲数; fold5_dir, 五折的目录; save_dir, 保存结果的目录;
    # output, avg_auc_aupr.txt, 大小为(实体个数, 2), 第1列是auc, 第2列是aupr信息;
    auc_aupr_times_bot = torch.zeros((ass_mat_shape[0], 3))
    # 由步骤类似roc_pr4folder, 省略注释
    for i in range(fold_num):
        # 加载数据
        eva_labels_outs_x_y = torch.from_numpy(np.loadtxt(os.path.join(fold5_dir, f'fold{i}.txt')))
        labels, pred_ys, test_x_ys = eva_labels_outs_x_y[:, 0], eva_labels_outs_x_y[:, 1], eva_labels_outs_x_y[:,
                                                                                           2: 4].to(torch.int)
        # 计算一折每个实体的auc和aupr
        labels_mat, pred_ys_mat, test_num = torch.zeros((ass_mat_shape)) - 1, torch.zeros((ass_mat_shape)) - 1, len(
            labels)
        for i in range(test_num):
            labels_mat[test_x_ys[i][0]][test_x_ys[i][1]] = labels[i]
            pred_ys_mat[test_x_ys[i][0]][test_x_ys[i][1]] = pred_ys[i]
        bool_mat4test = (labels_mat != -1)
        for i in range(ass_mat_shape[0]):
            if (labels_mat[i, :] == 1).sum() > 0:
                auc_aupr_times_bot[i, 2] += 1
                labels4test1rowi = labels_mat[i][bool_mat4test[i]]
                pred_y4test1rowi = pred_ys_mat[i][bool_mat4test[i]]
                fpr4rowi, tpr4rowi, _ = roc_curve(labels4test1rowi, pred_y4test1rowi)
                precision4rowi, recall4rowi, _ = precision_recall_curve(labels4test1rowi, pred_y4test1rowi)
                precision4rowi[-1] = [1, 0][(int)(precision4rowi[-2] == 0)]
                auc_aupr_times_bot[i, 0] += auc(fpr4rowi, tpr4rowi)
                auc_aupr_times_bot[i, 1] += auc(recall4rowi, precision4rowi)
    # 防止除数为0
    auc_aupr_times_bot[(auc_aupr_times_bot[:, 2] == 0), -1] = 1
    # 求avg
    auc_aupr_times_bot[:, 0] = (auc_aupr_times_bot[:, 0] * 1.0) / auc_aupr_times_bot[:, 2]
    auc_aupr_times_bot[:, 1] = (auc_aupr_times_bot[:, 1] * 1.0) / auc_aupr_times_bot[:, 2]
    # save
    auc_aupr = auc_aupr_times_bot[:, 0: 2]
    auc_value = auc_aupr_times_bot[:, 0]
    aupr_value = auc_aupr_times_bot[:, 1]
    np.savetxt(os.path.join(save_dir, 'avg_auc.txt'), auc_value)
    np.savetxt(os.path.join(save_dir, 'avg_aupr.txt'), aupr_value)


# wilcoxon实验
def wilcoxon_test4drug_auc_aupr(model_names, u_method_idx, data_dir):
    # input, model_names, a list with model names, e.g. ,['ngmda', 'scsmda', 'gsamda', 'gacnnmda', 'egatmda', 'gcnmda'];
    # u_method_idx, 0; 你模型于model_names列表的索引;
    # data_dir, 模型数据存放目录.
    u_method_avg_auc_aupr = np.loadtxt('mycode/CNN_disease/TestData/Wilcoxon/My/avg_auc_aupr.txt')
    for i in range(len(model_names)):
        if i != u_method_idx:
            auc = np.loadtxt(os.path.join(data_dir, model_names[i], 'avg_auc.txt'))
            aupr = np.loadtxt(os.path.join(data_dir, model_names[i], 'avg_aupr.txt'))
            print(f'{model_names[u_method_idx]}与 {model_names[i]} avg auc  wilcoxon test: {w_test(u_method_avg_auc_aupr[:, 0], auc)}')
            print(f'{model_names[u_method_idx]}与 {model_names[i]} avg aupr wilcoxon test: {w_test(u_method_avg_auc_aupr[:, 1], aupr)}')


if __name__ == '__main__':
    
    # # 由预测结果，坐标，标签，计算方法的AUCs AUPRs
    # # 首先将标签 预测结果 坐标按顺序存在txt文件中，预测结果取有关联的数，坐标先是药物在是微生物
    # _, _, _, _, test_index = torch.load('circ_CNN.pth')
    # # os.makedirs('mycode/CNN_disease/TestData/Wilcoxon/SGFCCDA', exist_ok=True)
    # for i in range(5):
    #     y = torch.load(rf'MLNGCFlabel{i}').to("cpu")
    #     pred = torch.load(rf'MLNGCFoutput{i}').to("cpu")
    #     # pred = torch.tensor(pred)
    #     pred = torch.softmax(pred, dim=1)[:, 1]
    #     test_index_i = test_index[i].T  # Transpose to match the dimensions
    #     # 合并标签，预测结果，坐标
    #     label_outs_index = torch.cat([y.unsqueeze(1), pred.unsqueeze(1), test_index_i], dim=1)
    #     np.savetxt(f'mycode/CNN_disease/TestData/Wilcoxon/xxx/fold{i}.txt', label_outs_index.numpy())

    # compute_methodA_avg_auc_aupr4every_entity((834, 138))
    
    # u = np.loadtxt('mycode/CNN_disease/TestData/Wilcoxon/BiSGTAR/avg_auc_aupr.txt')
    # aucv = u[:, 0]
    # auprv = u[:, 1]
    # np.savetxt(os.path.join("mycode/CNN_disease/TestData/Wilcoxon/xxx/", 'avg_auc.txt'), aucv)
    # np.savetxt(os.path.join("mycode/CNN_disease/TestData/Wilcoxon/xxx/", 'avg_aupr.txt'), auprv)
    
    # file_folder_name4avg_auc_aupr, 文件夹的名字(保存平均auc、aupr)
    models = ['My','BiSGTAR', 'GraphCDA', 'MDGF_MCEC', 'SGFCCDA', 'MLNGCF', 'MPCLCDA']
    # 对所有的模型的五折结果求avg auc与aupr
    # for i in range(len(models)):
    # 	compute_methodA_avg_auc_aupr4every_entity((1373, 173), 5, os.path.join('../compared_method/', models[i], 'fold5'), os.path.join('../compared_method/', models[i], file_folder_name4avg_auc_aupr))
    print('starting wilcoxon test...')
    # 进行wilcoxon_test
    wilcoxon_test4drug_auc_aupr(models, 0, 'mycode/CNN_disease/TestData/Wilcoxon/')

# MLNGCF=BiSGTAR>MPCLCDA>>GraphCDA>MDGF_MCEC=SGFCCDA
# BiSGTAR>GraphCDA>MDGF_MCEC>>SGFCCDA>>MLNGCF=MPCLCDA