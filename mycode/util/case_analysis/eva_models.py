from tools4roc_pr import roc_pr4_folder, roc_pr4cross_val
import numpy as np
import torch
def eva(folder_name):	
	mean_fpr_ls, mean_tpr_ls, mean_recall_ls, mean_prec_ls, fold_num= [], [], [], [], 5
	# 龙老师模型进行评估
	for i in range(fold_num):
		eva_labels_outs_x_y= torch.from_numpy(np.loadtxt(f'{folder_name}fold{i}.txt'))
		test_labels, preds, test_xy= eva_labels_outs_x_y[:, 0], eva_labels_outs_x_y[:, 1], eva_labels_outs_x_y[:, 2: 4].to(torch.int)
		mean_fpr, mean_tpr, mean_recall, mean_prec= roc_pr4_folder(test_xy, test_labels, preds, (1373, 173))
		mean_fpr_ls.append(torch.from_numpy(mean_fpr))
		mean_tpr_ls.append(torch.from_numpy(mean_tpr))
		mean_recall_ls.append(torch.from_numpy(mean_recall))
		mean_prec_ls.append(torch.from_numpy(mean_prec))
	mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts= torch.stack(mean_fpr_ls), torch.stack(mean_tpr_ls), torch.stack(mean_recall_ls), torch.stack(mean_prec_ls)
	roc_pr4cross_val(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, fold_num)

if __name__ == '__main__':
	# folder_name= 'E:/研究生生涯/论文事宜/01/消融实验end/model/fold5/'
	folder_name= 'F:/deeplearning/0214 EHGN4MDA-XIU/code/'
	# folder_name= 'F:/deeplearning/0228 mkgcn2/model result950807/'
	eva(folder_name)
