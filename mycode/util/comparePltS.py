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

def roc_pr_split(fpr, tpr, recall, prec, auc4roc, auc4pr, model, aboutAuc, aboutAUPR):
	fig, axes = plt.subplots(3, 4, figsize=(28, 20))  # Increased figure size
	fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Add space between rows and columns
	
	# Define custom RGB colors for different models
	colors = [(78/255, 171/255, 144/255), (122/255,27/255, 109/255), (115/255, 186/255, 214/255), (255/255,183/255, 3/255), (226/255, 157/255, 116/255), (107/255, 112/255, 92/255), (217/255, 79/255, 51/255)]
	
	# Plot ROC and PR Curves for each comparison
	for i, idx in enumerate([0, 2, 3, 4, 5, 6]):
		row = i // 2
		col = (i % 2) * 2
		
		# ROC Curve
		ax1 = axes[row, col]
		ax1.plot(fpr[1], tpr[1], color=colors[1], lw=2, alpha=0.8, label=model[1] + f' (AUC = {auc4roc[1]:.3f}±{aboutAuc[1]})')
		ax1.plot(fpr[idx], tpr[idx], color=colors[idx], lw=2, alpha=0.8, label=model[idx] + f' (AUC = {auc4roc[idx]:.3f}±{aboutAuc[idx]})')
		ax1.legend(loc='lower right', fontsize=18, bbox_to_anchor=(1.0, 0))  # Adjust legend position
		ax1.set_title(f'ROC Curve: {model[idx]} vs {model[1]}', fontsize=19)
		ax1.set_xlabel('FPR', fontsize=18)
		ax1.set_ylabel('TPR', fontsize=18)
		ax1.axis([-0.05, 1.05, -0.05, 1.05])  # Expanded axis range
		ax1.set_yticks(np.arange(0, 1.1, step=0.2))
		ax1.tick_params(axis='both', which='major', labelsize=18)  # Increased tick label font size
		
		# PR Curve
		ax2 = axes[row, col + 1]
		ax2.plot(recall[1], prec[1], color=colors[1], lw=2, alpha=0.8, label=model[1] + f' (AUPR = {auc4pr[1]:.3f}±{aboutAUPR[1]})')
		ax2.plot(recall[idx], prec[idx], color=colors[idx], lw=2, alpha=0.8, label=model[idx] + f' (AUPR = {auc4pr[idx]:.3f}±{aboutAUPR[idx]})')
		ax2.legend(loc='upper right', fontsize=18, bbox_to_anchor=(1.0, 1))  # Adjust legend position
		ax2.set_title(f'PR Curve: {model[idx]} vs {model[1]}', fontsize=19)
		ax2.set_xlabel('Recall', fontsize=18)
		ax2.set_ylabel('Precision', fontsize=18)
		ax2.axis([-0.05, 1.05, -0.05, 0.5])  # Expanded axis range
		ax2.tick_params(axis='both', which='major', labelsize=18)  # Increased tick label font size
	
	plt.tight_layout()
	plt.savefig('roc_pr_split_new.jpg', dpi=300, bbox_inches='tight')
	plt.show()


mean_fprs, mean_tprs, mean_recalls, mean_precs= [], [], [], []
_,c_d,fea,tri,test_data=torch.load('circ_CNN.pth')

mean_fpr_my, mean_tpr_my, mean_recall_my, mean_prec_my, mean_auc4roc_my, mean_auc4pr_my = torch.load('My_plt_s.pkl')
mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('SGFCCDA_plt_s.pkl')
fpr, tpr, recall, prec, auc4roc, auc4pr = [], [], [], [], [], []
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)
fpr.append(mean_fpr_my); tpr.append(mean_tpr_my); recall.append(mean_recall_my); prec.append(mean_prec_my); auc4roc.append(mean_auc4roc_my); auc4pr.append(mean_auc4pr_my)

mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('MLNGCF_plt_s.pkl')
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)

mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('MDGF_MCEC_plt_s.pkl')
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)
mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('BiSGTAR_plt_s.pkl')
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)
mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('GraphCDA_plt_s.pkl')
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)
mean_fpr, mean_tpr, mean_recall, mean_prec, mean_auc4roc, mean_auc4pr = torch.load('MPCLCDA_plt_s.pkl')
fpr.append(mean_fpr); tpr.append(mean_tpr); recall.append(mean_recall); prec.append(mean_prec); auc4roc.append(mean_auc4roc); auc4pr.append(mean_auc4pr)
# roc_pr_combine(fpr, tpr, recall, prec, auc4roc, auc4pr,['SGFCCDA', 'MKCD','MLNGCF','MDGF_MCEC','BiSGTAR', 'GraphCDA', 'MPCLCDA'])
roc_pr_split(fpr, tpr, recall, prec, auc4roc, auc4pr,['SGFCCDA', 'MKCD','MLNGCF','MDGF-MCEC','Bi-SGTAR', 'GraphCDA', 'MPCLCDA'], aboutAuc=[0.052, 0.007, 0.024, 0.067, 0.022, 0.056, 0.049], aboutAUPR=[0.021, 0.013, 0.005, 0.021, 0.017, 0.008, 0.006])
