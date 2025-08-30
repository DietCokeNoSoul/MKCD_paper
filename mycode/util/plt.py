import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

# function: 根据模型在测试集一折下的表现，计算A(主体)对不同B关联预测ROC的平均，例如disease与miRna的关联预测，则disease为主体,对应到ass_mat_shape,其维度是(disease的个数,miRna的个数).
# coder: 顾京
# update time: 22/09/07
# 参考资料：
# 1.https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
# 2.https://stackoverflow.com/questions/60865028/sklearn-precision-recall-curve-and-threshold
# 3.https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
# 注意：
# 以下变量命名中，4代表英文for，1代表英文in
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
			# print(labels4test1rowi)
			# print(pred_y4test1rowi)
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

	# 上述已经计算了所有主体A对不同B关联预测的ROC、PR
	# 下面开始插值求平均ROC、平均PR
	mean_fpr, mean_recall= np.linspace(0, 1, 100), np.linspace(0, 1, 100)
	# 此处是细节，我们知道对于某一条ROC，fpr定义域范围[x, 1], 当fpr= 1, tpr= 1；fpr=x，tpr=0；因此可以将fpr定义域范围延申为[0, 1],对延申部分，tpr值均为0填充，这是无影响的。
	# 我们知道对于某一条PR，recall定义范围[x, 1], 那么它能否也向左拉伸呢？
	# 	如果threshold取最大模型预测值时，该实例恰为负例，precision为0，recall为0.此时x=0；
	# 	若该实例恰为正例，precision为1，recall为tp/(tp+fn),tp= 1,fn= 剩余正例数，一般fn不会小，则recall比较小，接近0，但是此时还不是0，我们需要对其进行拉伸，向左延长precision值，即1（因为直觉上，threshold越大，即对正例越苛刻，查准率不会降低，查全率会降低，同时sklearn也是这么做的，但是没给出上述解释）。
	#	从而recall也是可以向左拉伸的。
	tpr_ls4mean_tpr, prec_ls4mean_prec= [], []
	for i in range(effective_rows_len):
		# 注意,np.interp必须保证x是递增的
		tpr_ls4mean_tpr.append(np.interp(mean_fpr, fpr_ls[i], tpr_ls[i]))
		prec_ls4mean_prec.append(np.interp(mean_fpr, recall_ls[i], prec_ls[i]))
	# 以上是插值
	mean_tpr, mean_prec= np.mean(tpr_ls4mean_tpr, axis= 0), np.mean(prec_ls4mean_prec, axis= 0)
	# 以上是求ROC的平均值
	# print(effective_rows_len)
	# 测试代码start
	# aucs4fpr_tpr, aucs4prec_recall= 0, 0
	# for i in range(effective_rows_len):
		# aucs4fpr_tpr+= auc(fpr_ls[i], tpr_ls[i])
		# aucs4prec_recall+= auc(recall_ls[i], prec_ls[i])
	# print(f'ROC平均值aucs4fpr_tpr/ effective_rows_len: {aucs4fpr_tpr/ effective_rows_len}')
	aucN = auc(mean_fpr, mean_tpr)
	auprN = auc(mean_recall, mean_prec)
	print(f'ROC平均值auc(mean_fpr, mean_tpr): {auc(mean_fpr, mean_tpr)}')
	# print(f'pr平均值aucs4prec_recall/ effective_rows_len: {aucs4prec_recall/ effective_rows_len}')
	print(f'pr平均值auc(mean_recall, mean_prec)：{auc(mean_recall, mean_prec)}')
	# 测试代码end	
	return mean_fpr, mean_tpr, mean_recall, mean_prec,aucN,auprN


# function: 对交叉验证的结果进行绘图 包括了每一折的ROC 平均的ROC
# coder: 顾京
# 注意：k_fold> 1, 该函数必须采用roc_pr4_folder函数计算mean_fpr, mean_tpr, mean_recall, mean_prec.否则不可直接按行求平均，即不可使用下函数。
def roc_pr4cross_val(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, k_fold):
	# mean_fpr_ts, dim: (k_fold, 100), 意义:每i元素就是第i折后的mean_fpr，由于mean_fpr是插值得到的，故而长度是100，由于是k折，故而维度是(k_fold, 100);
	# mean_tpr_ts, mean_recall_ts, mean_prec_ts同上, k_fold是标量 一共几折就是几。

	mean_fpr, mean_tpr, mean_recall, mean_prec= mean_fpr_ts[0], torch.mean(mean_tpr_ts, dim= 0), mean_recall_ts[0], torch.mean(mean_prec_ts, dim= 0)
	# 为什么fpr, recall不用按行求平均 因为他们每一行都是一样的（roc_pr4_folder函数里面是固定了fpr,recall对tpr,prec进行插值的，故而每一行一样） 求不求都一样
	# 下面就是简单的画图
	aucs4roc, aucs4pr= [], []
	for i in range(k_fold):
		aucs4roc.append(auc(mean_fpr_ts[i], mean_tpr_ts[i]))
		plt.plot(mean_fpr_ts[i], mean_tpr_ts[i], lw= 1, alpha= 0.3, label= f'ROC fold {i + 1} (AUC= {aucs4roc[i]:0.3f})')
  
	aucs4roc_std, mean_auc4roc= np.std(aucs4roc), auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, color= 'b', lw= 2, alpha= 0.8, label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc4roc, aucs4roc_std))
	plt.title('roc curve')
	plt.xlabel('fpr')
	plt.ylabel('tpr')
	plt.axis([0, 1, 0, 1])
	plt.legend(loc= 'lower right') 
	plt.savefig('roc_curve.png')  # 保存ROC曲线图像
	# plt.show()
	plt.clf()
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
	plt.savefig('pr_curve.png')  # 保存PR曲线图像
	# plt.show()
	plt.clf()
 

def roc_pr4cross_val_1(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, k_fold):
	mean_fpr, mean_tpr, mean_recall, mean_prec= mean_fpr_ts[0], torch.mean(mean_tpr_ts, dim= 0), mean_recall_ts[0], torch.mean(mean_prec_ts, dim= 0)
	torch.save([mean_fpr, mean_tpr, mean_recall, mean_prec],'BiSGTAR.pkl')
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
 
 
def plot_roc_pr_curves(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, k_fold):
	mean_fpr, mean_tpr, mean_recall, mean_prec = mean_fpr_ts[0], torch.mean(mean_tpr_ts, dim=0), mean_recall_ts[0], torch.mean(mean_prec_ts, dim=0)
	torch.save([mean_fpr, mean_tpr, mean_recall, mean_prec], 'BiSGTAR.pkl')
	aucs4roc, aucs4pr = [], []

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

	# Plot ROC curves
	for i in range(k_fold):
		aucs4roc.append(auc(mean_fpr_ts[i], mean_tpr_ts[i]))
		ax1.plot(mean_fpr_ts[i], mean_tpr_ts[i], lw=1, alpha=0.3, label='ROC fold %d (AUC= %0.3f)' % (i + 1, aucs4roc[i]))
	aucs4roc_std, mean_auc4roc = np.std(aucs4roc), auc(mean_fpr, mean_tpr)
	ax1.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc4roc, aucs4roc_std))
	ax1.set_title('ROC Curve')
	ax1.set_xlabel('FPR')
	ax1.set_ylabel('TPR')
	ax1.axis([0, 1, 0, 1])
	ax1.legend(loc='lower right')

	# Plot PR curves
	for i in range(k_fold):
		aucs4pr.append(auc(mean_recall_ts[i], mean_prec_ts[i]))
		ax2.plot(mean_recall_ts[i], mean_prec_ts[i], lw=1, alpha=0.3, label='PR fold %d (AUPR= %0.3f)' % (i + 1, aucs4pr[i]))
	aucs4pr_std, mean_auc4pr = np.std(aucs4pr), auc(mean_recall, mean_prec)
	ax2.plot(mean_recall, mean_prec, color='b', lw=2, alpha=0.8, label=r'Mean PR (AUPR = %0.3f $\pm$ %0.3f)' % (mean_auc4pr, aucs4pr_std))
	ax2.set_title('PR Curve')
	ax2.set_xlabel('Recall')
	ax2.set_ylabel('Precision')
	ax2.axis([0, 1, 0, 1])
	ax2.legend(loc='upper right')

	plt.tight_layout()
	plt.savefig('roc_pr_curve.png')
	plt.show()
