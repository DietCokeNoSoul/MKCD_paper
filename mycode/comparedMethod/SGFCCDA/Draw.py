import sys
import os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
import numpy as np
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

def roc_pr4_folder(test_x_ys, labels, pred_ys, ass_mat_shape):
	# test_x_ys, like: torch.tensor([[0, 0], [0, 1]....]), dim:(测试实例个数, 2), 意义:测试集里面实例在关联矩阵中的索引(下标)的集合;
	# ass_mat_shape, like: (a, b), dim: (1, 2), 意义:关联矩阵的维度；
	# labels, like: torch.tensor([0, 0, 1, 1, 0, 0, ....]), dim:(1,测试实例个数), 意义:测试集里面实例对应标签（与上面索引一一对应);
	# pred_ys, like: torch.tensor([0.012, 0.209, 0.8623, 0.98212, ...]), dim:(1,测试实例个数), 意义:模型对测试集里面实例预测关联的概率值；
	
	labels_mat, pred_ys_mat, test_num= torch.zeros((ass_mat_shape)) -1, torch.zeros((ass_mat_shape)) -1, len(labels)
	# labels_mat, 测试集的标签矩阵，其值-1表示该实例在训练集；pred_ys_mat，测试集的预测矩阵，其值-1表示该实例在训练集
	for i in range(test_num):
		labels_mat[test_x_ys[i][0], test_x_ys[i][1]]= labels[i]
		pred_ys_mat[test_x_ys[i][0], test_x_ys[i][1]]= pred_ys[i]
	# 对labels_mat，pred_ys_mat进行初始化
	bool_mat4test= (labels_mat!= -1)
	# bool_mat4test, (ass_mat_shape), 用于后面选测试的实例用
	fpr_ls, tpr_ls, recall_ls, prec_ls, effective_rows_len = [], [], [], [], 0
	for i in range(ass_mat_shape[0]):
		# 遍历的labels_mat，pred_ys_mat每一行 计算指标 
		if (labels_mat[i][bool_mat4test[i]]== 1).sum()> 0:
			effective_rows_len+= 1
			# 有正例方计算 如果没正例 ROC、PR曲线都是0 没意义 因此对其不进行计算 只要存在正例 计算ROC、PR就有了意义 
			labels4test1rowi= labels_mat[i][bool_mat4test[i]]
			pred_y4test1rowi= pred_ys_mat[i][bool_mat4test[i]]

			fpr4rowi, tpr4rowi, _= roc_curve(labels4test1rowi, pred_y4test1rowi)
			fpr_ls.append(fpr4rowi)
			tpr_ls.append(tpr4rowi)
			# 以上是计算某一主体A对不同B关联预测的ROC
			precision4rowi, recall4rowi, _= precision_recall_curve(labels4test1rowi, pred_y4test1rowi)
			# 加入下行是为了修正precision_recall_curve函数会自动添加一个点(0, 1)，问题在于，阈值最大可能会预测错误，此时recall= 0, precision= 0,对应(0, 0)点，没必要添加(0, 1)点了
			precision4rowi[-1]= [1, 0][precision4rowi[-2]== 0]
			# a[::-1]对列表进行逆置 因为后面插值需要横坐标递增
			prec_ls.append(precision4rowi[::-1])
			recall_ls.append(recall4rowi[::-1])

	mean_fpr, mean_recall= np.linspace(0, 1, 100), np.linspace(0, 1, 100)

	tpr_ls4mean_tpr, prec_ls4mean_prec= [], []
	for i in range(effective_rows_len):
		# 注意,np.interp必须保证x是递增的
		tpr_ls4mean_tpr.append(np.interp(mean_fpr, fpr_ls[i], tpr_ls[i]))
		prec_ls4mean_prec.append(np.interp(mean_fpr, recall_ls[i], prec_ls[i]))
	# 以上是插值
	mean_tpr, mean_prec= np.mean(tpr_ls4mean_tpr, axis= 0), np.mean(prec_ls4mean_prec, axis= 0)

	print(f'ROC平均值auc(mean_fpr, mean_tpr): {auc(mean_fpr, mean_tpr)}')
	# print(f'pr平均值aucs4prec_recall/ effective_rows_len: {aucs4prec_recall/ effective_rows_len}')
	print(f'pr平均值auc(mean_recall, mean_prec)：{auc(mean_recall, mean_prec)}')
	# 测试代码end	
	return mean_fpr, mean_tpr, mean_recall, mean_prec
def roc_pr4cross_val(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, k_fold):
	mean_fpr, mean_tpr, mean_recall, mean_prec= mean_fpr_ts[0], torch.mean(mean_tpr_ts, dim= 0), mean_recall_ts[0], torch.mean(mean_prec_ts, dim= 0)
	torch.save([mean_fpr,mean_tpr,mean_recall,mean_prec],'MLNGCF.pkl')
	aucs4roc, aucs4pr= [], []
	for i in range(k_fold):
		aucs4roc.append(auc(mean_fpr_ts[i], mean_tpr_ts[i]))
		plt.plot(mean_fpr_ts[i], mean_tpr_ts[i], lw= 1, alpha= 0.3, label= 'ROC fold %d (AUC= %0.3f)' % (i+ 1, aucs4roc[i]))
	aucs4roc_std, mean_auc4roc= np.std(aucs4roc), auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, color= 'b', lw= 2, alpha= 0.8, label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc4roc, aucs4roc_std))
	plt.title('roc curve')
	plt.xlabel('fpr')
	plt.ylabel('tpr')
	plt.axis([0, 1, 0, 1])
	plt.legend(loc= 'lower right')
	plt.show()
	for i in range(k_fold):
		aucs4pr.append(auc(mean_recall_ts[i], mean_prec_ts[i]))
		plt.plot(mean_recall_ts[i], mean_prec_ts[i], lw= 1, alpha= 0.3, label= 'PR fold %d (AUPR= %0.3f)' % (i+ 1, aucs4pr[i]))
	aucs4pr_std, mean_auc4pr= np.std(aucs4pr), auc(mean_recall, mean_prec)
	plt.plot(mean_recall, mean_prec, color= 'b', lw= 2, alpha= 0.8, label= r'Mean PR (AUPR = %0.3f $\pm$ %0.3f)' % (mean_auc4pr, aucs4pr_std))
	plt.title('pr curve')
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.axis([0, 1, 0, 1])
	plt.legend(loc= 'lower right')
	plt.show()
	torch.save([mean_fpr, mean_tpr, mean_recall, mean_prec,],'MLNGCF_plt.pkl')
 
mean_fprs, mean_tprs, mean_recalls, mean_precs= [], [], [], []
_,c_d,fea,tri,test_data=torch.load('circ_CNN.pth')
# 以疾病为主体
for i in range(5):
    y = torch.load(rf'mycode\comparedMethod\SGFCCDA\result_1\SGFCCDAlabel{i}').squeeze(1)
    pred = torch.load(rf'mycode\comparedMethod\SGFCCDA\result_1\SGFCCDAoutput{i}').squeeze(1)
    test_idx= test_data[i].T
    test_idx= torch.stack([test_idx[:, 1], test_idx[:, 0]], dim= 1)
    mean_fpr, mean_tpr, mean_recall, mean_prec= roc_pr4_folder(test_idx, y, pred, (138, 834))
    mean_fprs.append(torch.tensor(mean_fpr)); mean_tprs.append(torch.tensor(mean_tpr)); mean_recalls.append(torch.tensor(mean_recall)); mean_precs.append(torch.tensor(mean_prec))
mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts= torch.stack(mean_fprs), torch.stack(mean_tprs), torch.stack(mean_recalls, dim= 0), torch.stack(mean_precs, dim= 0)
torch.save([mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts],'SGFCCDA.pkl')

# roc_pr4cross_val(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, 5)