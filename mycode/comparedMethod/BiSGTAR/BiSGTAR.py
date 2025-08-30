import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

class TAR(nn.Module):
    def __init__(self, hid_dim, out_dim, bias=False):
        super(TAR, self).__init__()
        # encoder-1
        self.e1 = nn.Linear(out_dim, hid_dim, bias=bias)
        # decoder
        self.d1 = nn.Linear(hid_dim, out_dim, bias=bias)
        self.Confidence = nn.Linear(hid_dim, out_dim, bias=bias)
        self.act1 = nn.ELU()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.Sigmoid()
    def encoder(self, x):
        h = self.act1(self.e1(x))
        return h
    def decoder(self, z):
        h = self.act2(self.d1(z))
        return h
    def confidencer(self, z):
        y = self.act3(self.Confidence(z))
        return y
    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder(z)
        y = self.confidencer(z)
        return y, h
class BiSGTAR(nn.Module):
    def __init__(self, args):
        super(BiSGTAR, self).__init__()
        dis_num = args.dis_num
        rna_num = args.rna_num
        self.input_drop = nn.Dropout(0.)
        self.att_drop = nn.Dropout(0.)
        self.FeatQC_rna = nn.Linear(dis_num, dis_num, bias=True)
        self.FeatQC_dis = nn.Linear(rna_num, rna_num, bias=True)
        self.AE_rna = TAR(args.hidden, dis_num)
        self.AE_dis = TAR(args.hidden, rna_num)
        self.act = nn.Sigmoid()
        self.dropout = args.dropout
    def forward(self, feat):
        rna_quality = self.act(F.dropout(self.FeatQC_rna(feat), self.dropout))
        dis_quality = self.act(F.dropout(self.FeatQC_dis(feat.t()), self.dropout))
        rna_sparse_feat = torch.mul(rna_quality, feat)
        dis_sparse_feat = torch.mul(dis_quality, feat.t())
        yc, hc = self.AE_rna(rna_sparse_feat)
        yd, hd = self.AE_dis(dis_sparse_feat)
        return yc, rna_sparse_feat, rna_quality, hc, yd, dis_sparse_feat, dis_quality, hd
# feat=torch.randn(834,138)
# class param():
#     def __init__(self):
#         self.dis_num=138
#         self.rna_num=834
#         self.hidden=64
#         self.dropout=0.5
# args=param()
# BiSGTAR(args)(feat)[0].shape

_,cd,fea,tri,tei=torch.load('circ_CNN.pth')

class param():
    def __init__(self):
        self.dis_num=138
        self.rna_num=834
        self.hidden=64
        self.dropout=0.5
args=param()
res=[]
criterion = nn.BCELoss()
for i in range(5):
    print('cross:%d'%i)
    net=BiSGTAR(args).to(device)
    optimizer=torch.optim.Adam(net.parameters(),8e-3,weight_decay=1e-10)
    feat=fea[i][:834,834:834+138].float().to(device)
    starttime = time.time()
    for e in range(600):
        yl, rna_feat, rna_quality, hc, yd, dis_feat, dis_quality, hd=net(feat)
        y = 0.5 * yl + (1 - 0.5) * yd.t()
        rna_confidence = torch.mul(hc, feat)
        dis_confidence = torch.mul(hd, feat.t())
        rna_SPC = torch.mean(rna_quality)
        rna_TAR = criterion(hc, feat) + F.mse_loss(yl, rna_confidence)
        rna_loss = 0.8 * rna_TAR + 0.2 * rna_SPC
        dis_SPC = torch.mean(dis_quality)
        dis_TAR = criterion(hd, feat.t()) + F.mse_loss(yd, dis_confidence)
        dis_loss = 0.8 * dis_TAR + 0.2 * dis_SPC
        loss_inter = 0.8 * rna_loss + 0.2 * dis_loss
        loss_cls = criterion(y, feat)
        loss = 0.8* loss_cls + 0.2 * loss_inter
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    endtime = time.time()
    print(endtime-starttime)
    yl, rna_feat, rna_quality, hc, yd, dis_feat, dis_quality, hd=net(feat)
    y = 0.5 * yl + (1 - 0.5) * yd.t()
    res.append([y[tei[i][0,:],tei[i][1,:]].detach().cpu(),cd[tei[i][0,:],tei[i][1,:]]])
    

# def roc_pr4_folder(test_x_ys, labels, pred_ys, ass_mat_shape):
# 	labels_mat, pred_ys_mat, test_num= torch.zeros((ass_mat_shape)) -1, torch.zeros((ass_mat_shape)) -1, len(labels)
# 	for i in range(test_num):
# 		labels_mat[test_x_ys[i][0], test_x_ys[i][1]]= labels[i]
# 		pred_ys_mat[test_x_ys[i][0], test_x_ys[i][1]]= pred_ys[i]
# 	bool_mat4test= (labels_mat!= -1)
# 	fpr_ls, tpr_ls, recall_ls, prec_ls, effective_rows_len = [], [], [], [], 0
# 	for i in range(ass_mat_shape[0]):
# 		if (labels_mat[i][bool_mat4test[i]]== 1).sum()> 0:
# 			effective_rows_len+= 1
# 			labels4test1rowi= labels_mat[i][bool_mat4test[i]]
# 			pred_y4test1rowi= pred_ys_mat[i][bool_mat4test[i]]
# 			fpr4rowi, tpr4rowi, _= roc_curve(labels4test1rowi, pred_y4test1rowi)
# 			fpr_ls.append(fpr4rowi)
# 			tpr_ls.append(tpr4rowi)
# 			precision4rowi, recall4rowi, _= precision_recall_curve(labels4test1rowi, pred_y4test1rowi)
# 			precision4rowi[-1]= [1, 0][precision4rowi[-2]== 0]
# 			prec_ls.append(precision4rowi[::-1])
# 			recall_ls.append(recall4rowi[::-1])
# 	mean_fpr, mean_recall= np.linspace(0, 1, 100), np.linspace(0, 1, 100)
# 	tpr_ls4mean_tpr, prec_ls4mean_prec= [], []
# 	for i in range(effective_rows_len):
# 		tpr_ls4mean_tpr.append(np.interp(mean_fpr, fpr_ls[i], tpr_ls[i]))
# 		prec_ls4mean_prec.append(np.interp(mean_fpr, recall_ls[i], prec_ls[i]))
# 	mean_tpr, mean_prec= np.mean(tpr_ls4mean_tpr, axis= 0), np.mean(prec_ls4mean_prec, axis= 0)
# 	aucN = auc(mean_fpr, mean_tpr)
# 	auprN = auc(mean_recall, mean_prec)
# 	print(f'ROC平均值auc(mean_fpr, mean_tpr): {auc(mean_fpr, mean_tpr)}')
# 	print(f'pr平均值auc(mean_recall, mean_prec)：{auc(mean_recall, mean_prec)}')
# 	return mean_fpr, mean_tpr, mean_recall, mean_prec, aucN, auprN

# def roc_pr4cross_val(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, k_fold):
# 	mean_fpr, mean_tpr, mean_recall, mean_prec= mean_fpr_ts[0], torch.mean(mean_tpr_ts, dim= 0), mean_recall_ts[0], torch.mean(mean_prec_ts, dim= 0)
# 	torch.save([mean_fpr, mean_tpr, mean_recall, mean_prec],'BiSGTAR.pkl')
# 	aucs4roc, aucs4pr= [], []
# 	for i in range(k_fold):
# 		aucs4roc.append(auc(mean_fpr_ts[i], mean_tpr_ts[i]))
# 		plt.plot(mean_fpr_ts[i], mean_tpr_ts[i], lw= 1, alpha= 0.3, label= 'ROC fold %d (AUC= %0.3f)' % (i+ 1, aucs4roc[i]))
# 	aucs4roc_std, mean_auc4roc= np.std(aucs4roc), auc(mean_fpr, mean_tpr)
# 	plt.plot(mean_fpr, mean_tpr, color= 'b', lw= 2, alpha= 0.8, label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc4roc, aucs4roc_std))
# 	plt.title('roc curve')
# 	plt.xlabel('fpr')
# 	plt.ylabel('tpr')
# 	plt.axis([0, 1, 0, 1])
# 	plt.legend(loc= 'lower right')
# 	plt.show()
# 	for i in range(k_fold):
# 		aucs4pr.append(auc(mean_recall_ts[i], mean_prec_ts[i]))
# 		plt.plot(mean_recall_ts[i], mean_prec_ts[i], lw= 1, alpha= 0.3, label= 'PR fold %d (AUPR= %0.3f)' % (i+ 1, aucs4pr[i]))
# 	aucs4pr_std, mean_auc4pr= np.std(aucs4pr), auc(mean_recall, mean_prec)
# 	plt.plot(mean_recall, mean_prec, color= 'b', lw= 2, alpha= 0.8, label= r'Mean PR (AUPR = %0.3f $\pm$ %0.3f)' % (mean_auc4pr, aucs4pr_std))
# 	plt.title('pr curve')
# 	plt.xlabel('recall')
# 	plt.ylabel('precision')
# 	plt.axis([0, 1, 0, 1])
# 	plt.legend(loc= 'lower right')
# 	plt.show()
 
# mean_auc, mean_aupr= [], []
# mean_fprs, mean_tprs, mean_recalls, mean_precs= [], [], [], []
# _,cd,fea,tri,tei=torch.load('circ_CNN.pth')

# for i in range(5):
#     # pred, y=torch.load('./data_circ/final_model/circ_plt_%d'%i)
#     pred, y=res[i]
#     test_idx= tei[i].T
#     test_idx= torch.stack([test_idx[:, 1], test_idx[:, 0]], dim= 1)
#     mean_fpr, mean_tpr, mean_recall, mean_prec,aucN,auprN= roc_pr4_folder(test_idx, y, pred, (138, 834))
#     mean_auc.append(aucN)
#     mean_aupr.append(auprN)
#     mean_fprs.append(torch.tensor(mean_fpr)); mean_tprs.append(torch.tensor(mean_tpr)); mean_recalls.append(torch.tensor(mean_recall)); mean_precs.append(torch.tensor(mean_prec))
# # 打印auc和aupr的平均值
# print("Mean AUC: ", torch.mean(torch.tensor(mean_auc)))
# print("Mean AUPR: ", torch.mean(torch.tensor(mean_aupr)))
# mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts= torch.stack(mean_fprs), torch.stack(mean_tprs), torch.stack(mean_recalls, dim= 0), torch.stack(mean_precs, dim= 0)
# roc_pr4cross_val(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, 5)