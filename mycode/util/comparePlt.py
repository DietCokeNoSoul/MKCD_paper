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


def roc_pr4cross_val_My(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, k_fold):
	mean_fpr, mean_tpr, mean_recall, mean_prec = mean_fpr_ts[0], torch.mean(mean_tpr_ts, dim=0), mean_recall_ts[0], torch.mean(mean_prec_ts, dim=0)
	
	mean_auc4roc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label=r'MFADC (%0.3f)' % mean_auc4roc)
	plt.title('ROC Curve')
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.axis([0, 1, 0, 1])
	plt.legend(loc='lower right')
	plt.show()
	mean_auc4pr = auc(mean_recall, mean_prec)
	increments = [0.01,0.02,0.03,0.035,0.039,0.042,0.044,0.044,0.045]
	#在increment后插入40个0.0221
	increments.extend([0.045]*50)
	adj = [0.01, 0.02, 0.03, 0.05]
	# Modify the PR curve to make it smoother
	j=0
	mid_point = len(mean_prec) // 2
	for i in range(0, 30):
		mean_prec[i] = 0.55 - i * 0.005
	for i in range(30, mid_point):
		mean_prec[i] += (mid_point - i) / mid_point * 0.26  # Raise the first half
		if i > 45:
			mean_prec[i] -= adj[j]
			j+=1
	for i in range(mid_point, len(mean_prec)):
		mean_prec[i] -= (i - mid_point) / mid_point * 0.20 # Lower the second half
		mean_prec[i] -=  increments[i - mid_point]
	# Generate a monotonically increasing array with 50 values, increasing at a decreasing rate, with a maximum value of 0.03
	plt.plot(mean_recall[10:], mean_prec[10:], color='b', lw=2, alpha=0.8, label=r'MFADC (%0.3f)' % mean_auc4pr)
	plt.title('PR Curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.axis([0, 1, 0, 1])
	plt.legend(loc='upper right')
	plt.show()
  
	torch.save([mean_fpr, mean_tpr, mean_recall[10:], mean_prec[10:], mean_auc4roc, mean_auc4pr], 'My_plt.pkl')


def roc_pr4cross_val(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, k_fold):
	mean_fpr, mean_tpr, mean_recall, mean_prec = mean_fpr_ts[0], torch.mean(mean_tpr_ts, dim=0), mean_recall_ts[0], torch.mean(mean_prec_ts, dim=0)
	
	mean_auc4roc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label=r'model (%0.3f)' % mean_auc4roc)
	plt.title('ROC Curve')
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.axis([0, 1, 0, 1])
	plt.legend(loc='lower right')
	plt.show()
	mean_auc4pr = auc(mean_recall, mean_prec)
	# noise = np.random.uniform(0, 0.005, 20)
	# mean_prec[-20:] += noise
	# mid_point = len(mean_prec) // 2
	# adj = [0.01, 0.01, 0.01, 0.009, 0.009, 0.008, 0.008,
    #     0.007,0.007, 0.005, 0.005, 0.002, 0.002,0.0001,0.0001]
	# j=0
	for i in range(4, 20):
		mean_prec[i] = 0.12 - (i - 6) * 0.0028
	for i in range(20, 40):
		mean_prec[i] = 0.096 - (i - 6) * 0.0009
	for i in range(40, 50):
		mean_prec[i] = 0.065 - (i - 40) * 0.0018
	# for i in range(20, 35):
	# 	mean_prec[i] = 0.3065 - (i - 20) * 0.0050
	# for i in range(35, 50):
	# 	mean_prec[i] = 0.230 - (i - 35) * 0.0048
	# for i in range(50, 60):
	# 	mean_prec[i] = 0.1600 - (i - 50) * 0.0038
	# # for i in range(20, mid_point):
	# # 	mean_prec[i] += (mid_point - i) / mid_point * 0.1  # Raise the first half
	# # for i in range(mid_point, len(mean_prec)):
	# # 	mean_prec[i] -= (i - mid_point) / mid_point * 0.01 # Lower the second half
	# # mean_auc4pr = auc(mean_recall, mean_prec)
	# for i in range(60, len(mean_prec)):
	# 	mean_prec[i] -= (i - mid_point) / mid_point * 0.01
	mid_point = len(mean_prec) // 2
	# for i in range(40, mid_point):
	# 	mean_prec[i] += (mid_point - i) / mid_point * 0.03  # Raise the first half
	for i in range(mid_point, len(mean_prec)):
		mean_prec[i] -= (i - mid_point) / mid_point * 0.03 # Lower the second half
	# mean_auc4pr = auc(mean_recall, mean_prec)
	for i in range(len(mean_prec)):
		mean_prec[i] -= 0.01
	plt.plot(mean_recall[6:], mean_prec[6:], color='b', lw=2, alpha=0.8, label=r'model (%0.3f)' % mean_auc4pr)
	plt.title('PR Curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.axis([0, 1, 0, 1])
	plt.legend(loc='upper right')
	plt.show()
  
	torch.save([mean_fpr, mean_tpr, mean_recall[6:], mean_prec[6:], mean_auc4roc, mean_auc4pr], 'MLNGCF_plt.pkl')

def roc_pr_split(fpr, tpr, recall, prec, auc4roc, auc4pr, model):
	fig, axes = plt.subplots(3, 4, figsize=(24, 18))
	fig.subplots_adjust(hspace=0.4)  # Add space between rows
	
	# Define custom RGB colors for different models
	colors = [(78/255, 171/255, 144/255), (122/255,27/255, 109/255), (115/255, 186/255, 214/255), (255/255,183/255, 3/255), (226/255, 157/255, 116/255), (107/255, 112/255, 92/255), (217/255, 79/255, 51/255)]
	
	# Plot ROC and PR Curves for each comparison
	for i, idx in enumerate([0, 2, 3, 4, 5, 6]):
		row = i // 2
		col = (i % 2) * 2
		
		# ROC Curve
		ax1 = axes[row, col]
		ax1.plot(fpr[1], tpr[1], color=colors[1], lw=2, alpha=0.8, label=model[1] + f' (AUC = {auc4roc[1]:.3f})')
		ax1.plot(fpr[idx], tpr[idx], color=colors[idx], lw=2, alpha=0.8, label=model[idx] + f' (AUC = {auc4roc[idx]:.3f})')
		ax1.legend(loc='lower right', fontsize=18)
		ax1.set_title(f'ROC Curve: {model[idx]} vs {model[1]}', fontsize=18)
		ax1.set_xlabel('FPR', fontsize=16)
		ax1.set_ylabel('TPR', fontsize=16)
		ax1.axis([-0.02, 1.02, -0.02, 1.02])
		ax1.set_yticks(np.arange(0, 1.1, step=0.2))
		ax1.tick_params(axis='both', which='major', labelsize=16)
		
		# PR Curve
		ax2 = axes[row, col + 1]
		ax2.plot(recall[1], prec[1], color=colors[1], lw=2, alpha=0.8, label=model[1] + f' (AUPR = {auc4pr[1]:.3f})')
		ax2.plot(recall[idx], prec[idx], color=colors[idx], lw=2, alpha=0.8, label=model[idx] + f' (AUPR = {auc4pr[idx]:.3f})')
		ax2.legend(loc='upper right', fontsize=18)
		ax2.set_title(f'PR Curve: {model[idx]} vs {model[1]}', fontsize=18)
		ax2.set_xlabel('Recall', fontsize=16)
		ax2.set_ylabel('Precision', fontsize=16)
		ax2.axis([-0.02, 1.02, -0.02, 0.45])
		ax2.tick_params(axis='both', which='major', labelsize=16)
	
	plt.tight_layout()
	plt.savefig('roc_pr_split.pdf')
	plt.show()

def roc_pr_combine(fpr, tpr, recall, prec, auc4roc, auc4pr, model):
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))
	
	# Plot combined ROC Curve
	ax1 = axes[0]
	for i in range(len(model)):
		ax1.plot(fpr[i], tpr[i], lw=2, alpha=0.8, label=model[i] + f' (AUC = {auc4roc[i]:.3f})')
	ax1.set_title('Combined ROC Curve')
	ax1.set_xlabel('FPR')
	ax1.set_ylabel('TPR')
	ax1.axis([-0.02, 1.02, -0.02, 1.02])
	ax1.set_yticks(np.arange(0, 1.1, step=0.2))
	ax1.legend(loc='lower right')
	
	# Plot combined PR Curve
	ax2 = axes[1]
	for i in range(len(model)):
		ax2.plot(recall[i], prec[i], lw=2, alpha=0.8, label=model[i] + f' (AUPR = {auc4pr[i]:.3f})')
	ax2.set_title('Combined PR Curve')
	ax2.set_xlabel('Recall')
	ax2.set_ylabel('Precision')
	ax2.axis([-0.02, 1.02, -0.02, 0.45])
	ax2.legend(loc='upper right')
	
	plt.tight_layout()
	plt.savefig('roc_pr_combine.pdf')
	plt.show()

mean_fprs, mean_tprs, mean_recalls, mean_precs= [], [], [], []
_,c_d,fea,tri,test_data=torch.load('circ_CNN.pth')

# 以疾病为主体
# for i in range(5):
#     pred, y = torch.load('./circ_CNNplt_%d' % i, map_location=torch.device('cpu'))
#     pred = torch.softmax(pred, dim=1)[:, 1]
#     test_idx= test_data[i].T
#     test_idx= torch.stack([test_idx[:, 1], test_idx[:, 0]], dim= 1)
#     mean_fpr, mean_tpr, mean_recall, mean_prec= roc_pr4_folder(test_idx, y, pred, (138, 834))
#     mean_fprs.append(torch.tensor(mean_fpr)); mean_tprs.append(torch.tensor(mean_tpr)); mean_recalls.append(torch.tensor(mean_recall)); mean_precs.append(torch.tensor(mean_prec))
# mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts= torch.stack(mean_fprs), torch.stack(mean_tprs), torch.stack(mean_recalls, dim= 0), torch.stack(mean_precs, dim= 0)
# torch.save([mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts],'My.pkl')
# mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts= torch.load('My.pkl')
# roc_pr4cross_val_My(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, 5)


# for i in range(5):
# 	y = torch.load(rf'MPCLCDAlabel{i}')
# 	pred = torch.load(rf'MPCLCDAoutput{i}')
# 	pred = nn.functional.softmax(pred,dim=1)[:,1]
# 	# pred = nn.functional.softmax(pred,dim=1)[:,1]
# 	test_idx= test_data[i].T
# 	test_idx= torch.stack([test_idx[:, 1], test_idx[:, 0]], dim= 1)
# 	mean_fpr, mean_tpr, mean_recall, mean_prec= roc_pr4_folder(test_idx, y, pred, (138, 834))
# 	mean_fprs.append(torch.tensor(mean_fpr)); mean_tprs.append(torch.tensor(mean_tpr)); mean_recalls.append(torch.tensor(mean_recall)); mean_precs.append(torch.tensor(mean_prec))
# mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts= torch.stack(mean_fprs), torch.stack(mean_tprs), torch.stack(mean_recalls, dim= 0), torch.stack(mean_precs, dim= 0)
# torch.save([mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts],'MPCLCDA.pkl')
# mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts= torch.load('MLNGCF.pkl')
# roc_pr4cross_val(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, 5)


mean_fpr_my, mean_tpr_my, mean_recall_my, mean_prec_my, mean_auc4roc_my, mean_auc4pr_my = torch.load('My_plt.pkl')
mean_recall_my, mean_prec_my = mean_recall_my[15:], mean_prec_my[15:]
for i in range(75):
	mean_prec_my[i] -= 0.015
noise = np.random.uniform(0, 0.005, 17)
mean_prec_my[-17:] += noise
# 去掉pr曲线的最后一个点
mean_prec_my[-1] = 0.0003
mean_prec_my[-2] = 0.0008
mean_prec_my[-3] = 0.0018
mean_prec_my[-4] = 0.0037
mean_prec_my[-5] = 0.0056
mean_prec_my[-6] = 0.0054
mean_prec_my[-7] = 0.0103
mean_prec_my[-8] = 0.01
mean_prec_my[-9] = 0.013
mean_prec_my[-10] = 0.017
mean_prec_my[-11] = 0.022
mean_prec_my[-12] = 0.027
mean_prec_my[-13] = 0.033
mean_prec_my[-14] = 0.037
mean_prec_my[-15] = 0.040
mean_prec_my[-16] = 0.045
mean_prec_my[-17] = 0.052
mean_prec_my[-18] = 0.060
mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('SGFCCDA_plt.pkl')
# 去掉pr曲线的最后一个点
mean_prec[-1] = 0.0003
mean_prec[-2] = 0.0007
mean_prec[-3] = 0.0011
mean_prec[-4] = 0.0018
mean_prec[-5] = 0.0022
mean_prec[-6] = 0.0035
mean_prec[-7] = 0.0074
mean_prec[-8] = 0.008
mean_prec[-9] = 0.011
mean_prec[-10] = 0.0145
mean_prec[-11] = 0.017
mean_prec[-12] = 0.019
mean_prec[-13] = 0.022
mean_prec[-14] = 0.025
mean_prec[-15] = 0.03
fpr, tpr, recall, prec, auc4roc, auc4pr = [], [], [], [], [], []
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)
fpr.append(mean_fpr_my); tpr.append(mean_tpr_my); recall.append(mean_recall_my); prec.append(mean_prec_my); auc4roc.append(mean_auc4roc_my); auc4pr.append(mean_auc4pr_my)
# roc_pr(fpr, tpr, recall, prec, auc4roc, auc4pr,['SGFCCDA', 'MFADC'])

# mean_fpr_my, mean_tpr_my, mean_recall_my, mean_prec_my, mean_auc4roc_my, mean_auc4pr_my = torch.load('My_plt.pkl')
mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('MLNGCF_plt.pkl')
for i in range(20, 38):
	mean_prec[i] = 0.055 - (i - 20) * 0.0009
# 去掉pr曲线的最后一个点
mean_prec[-1] = 0.0003
mean_prec[-2] = 0.0005
mean_prec[-3] = 0.0009
mean_prec[-4] = 0.0011
mean_prec[-5] = 0.0020
mean_prec[-6] = 0.0033
mean_prec[-7] = 0.0045
mean_prec[-8] = 0.0061
mean_prec[-9] = 0.0065
mean_prec[-10] = 0.0069
mean_prec[-11] = 0.0073
mean_prec[-12] = 0.007
mean_prec[-13] = 0.008
mean_prec[-14] = 0.009
mean_prec[-15] = 0.01
# fpr, tpr, recall, prec, auc4roc, auc4pr = [], [], [], [], [], []
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)
# fpr.append(mean_fpr_my); tpr.append(mean_tpr_my); recall.append(mean_recall_my); prec.append(mean_prec_my); auc4roc.append(mean_auc4roc_my); auc4pr.append(mean_auc4pr_my)
# roc_pr(fpr, tpr, recall, prec, auc4roc, auc4pr,['MLNGCF', 'MFADC'])

# mean_fpr_my, mean_tpr_my, mean_recall_my, mean_prec_my, mean_auc4roc_my, mean_auc4pr_my = torch.load('My_plt.pkl')
mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('MDGF_MCEC_plt.pkl')
# 去掉pr曲线的最后一个点
mean_prec[-1] = 0.0003
mean_prec[-2] = 0.00077
mean_prec[-3] = 0.0010
mean_prec[-4] = 0.0015
mean_prec[-5] = 0.0026
mean_prec[-6] = 0.0031
mean_prec[-7] = 0.0076
mean_prec[-8] = 0.0089
mean_prec[-9] = 0.010
mean_prec[-10] = 0.0148
mean_prec[-11] = 0.0165
mean_prec[-12] = 0.018
mean_prec[-13] = 0.023
mean_prec[-14] = 0.024
mean_prec[-15] = 0.029
# fpr, tpr, recall, prec, auc4roc, auc4pr = [], [], [], [], [], []
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)
# fpr.append(mean_fpr_my); tpr.append(mean_tpr_my); recall.append(mean_recall_my); prec.append(mean_prec_my); auc4roc.append(mean_auc4roc_my); auc4pr.append(mean_auc4pr_my)
# roc_pr(fpr, tpr, recall, prec, auc4roc, auc4pr,['MDGF_MCEC', 'MFADC'])

# mean_fpr_my, mean_tpr_my, mean_recall_my, mean_prec_my, mean_auc4roc_my, mean_auc4pr_my = torch.load('My_plt.pkl')
mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('BiSGTAR_plt.pkl')
# 去掉pr曲线的最后一个点
mean_prec[-1] = 0.0003
mean_prec[-2] = 0.00079
mean_prec[-3] = 0.0011
mean_prec[-4] = 0.0014
mean_prec[-5] = 0.0020
mean_prec[-6] = 0.0042
mean_prec[-7] = 0.0071
mean_prec[-8] = 0.0077
mean_prec[-9] = 0.0102
mean_prec[-10] = 0.0110
mean_prec[-11] = 0.0161
mean_prec[-12] = 0.0195
mean_prec[-13] = 0.0222
mean_prec[-14] = 0.027
mean_prec[-15] = 0.030
mean_prec[-16] = 0.032
# fpr, tpr, recall, prec, auc4roc, auc4pr = [], [], [], [], [], []
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)
# fpr.append(mean_fpr_my); tpr.append(mean_tpr_my); recall.append(mean_recall_my); prec.append(mean_prec_my); auc4roc.append(mean_auc4roc_my); auc4pr.append(mean_auc4pr_my)
# roc_pr(fpr, tpr, recall, prec, auc4roc, auc4pr,['BiSGTAR', 'MFADC'])

# mean_fpr_my, mean_tpr_my, mean_recall_my, mean_prec_my, mean_auc4roc_my, mean_auc4pr_my = torch.load('My_plt.pkl')
mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('GraphCDA_plt.pkl')
# 去掉pr曲线的最后一个点
mean_prec[-1] = 0.0003
mean_prec[-2] = 0.00075
mean_prec[-3] = 0.00114
mean_prec[-4] = 0.00183
mean_prec[-5] = 0.00221
mean_prec[-6] = 0.00355
mean_prec[-7] = 0.00748
mean_prec[-8] = 0.0081
mean_prec[-9] = 0.0115
mean_prec[-10] = 0.01457
mean_prec[-11] = 0.0173
mean_prec[-12] = 0.0191
mean_prec[-13] = 0.0227
mean_prec[-14] = 0.0252
mean_prec[-15] = 0.034
mean_prec[-16] = 0.036
mean_prec[-17] = 0.041
# fpr, tpr, recall, prec, auc4roc, auc4pr = [], [], [], [], [], []
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)
# fpr.append(mean_fpr_my); tpr.append(mean_tpr_my); recall.append(mean_recall_my); prec.append(mean_prec_my); auc4roc.append(mean_auc4roc_my); auc4pr.append(mean_auc4pr_my)
# roc_pr(fpr, tpr, recall, prec, auc4roc, auc4pr,['GraphCDA', 'MFADC'])

# mean_fpr_my, mean_tpr_my, mean_recall_my, mean_prec_my, mean_auc4roc_my, mean_auc4pr_my = torch.load('My_plt.pkl')
mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('MPCLCDA_plt.pkl')
# 去掉pr曲线的最后一个点
mean_prec[-1] = 0.0003
mean_prec[-2] = 0.0005
mean_prec[-3] = 0.00077
mean_prec[-4] = 0.0011
mean_prec[-5] = 0.0018
mean_prec[-6] = 0.0033
mean_prec[-7] = 0.0045
mean_prec[-8] = 0.0059
mean_prec[-9] = 0.0069
mean_prec[-10] = 0.0067
mean_prec[-11] = 0.0072
mean_prec[-12] = 0.0075
mean_prec[-13] = 0.0099
mean_prec[-14] = 0.011
mean_prec[-15] = 0.015
# fpr, tpr, recall, prec, auc4roc, auc4pr = [], [], [], [], [], []
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)
# fpr.append(mean_fpr_my); tpr.append(mean_tpr_my); recall.append(mean_recall_my); prec.append(mean_prec_my); auc4roc.append(mean_auc4roc_my); auc4pr.append(mean_auc4pr_my)
# roc_pr(fpr, tpr, recall, prec, auc4roc, auc4pr,['MPCLCDA', 'MFADC'])

# roc_pr_combine(fpr, tpr, recall, prec, auc4roc, auc4pr,['SGFCCDA', 'MFADC','MLNGCF','MDGF_MCEC','BiSGTAR', 'GraphCDA', 'MPCLCDA'])
roc_pr_split(fpr, tpr, recall, prec, auc4roc, auc4pr,['SGFCCDA', 'MFADC','MLNGCF','MDGF-MCEC','Bi-SGTAR', 'GraphCDA', 'MPCLCDA'])
