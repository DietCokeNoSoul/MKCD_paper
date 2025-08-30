import matplotlib.pyplot as plt
import numpy as np
from numpy import argsort, array, average, ndarray, stack, sum, vstack, zeros
from scipy.stats import ranksums,wilcoxon
from sklearn.metrics import auc

from compare import get_result

plt.rc('font',family='Times New Roman')

def confusion(score:ndarray,label:ndarray):
    TP,FP,TN,FN=0,1,2,3
    ind=argsort(score)
    score=np.take_along_axis(score,ind,axis=0)
    label=np.take_along_axis(label,ind,axis=0)
    c=np.zeros((len(score)+1,4))
    tp=label.sum()
    fp=len(score)-tp
    tn,fn=0,0
    c[0,TP]=tp
    c[0,FP]=fp
    for i,l in enumerate(label):
        if l==1:
            tp-=1
            fn+=1
        else:
            tn+=1
            fp-=1
        c[i+1]=[tp,fp,tn,fn]
    return c

def confusion_by_threshold(score:ndarray,label:ndarray,thresholds:ndarray):
    ind=argsort(score)
    score=np.take_along_axis(score,ind,axis=0)
    label=np.take_along_axis(label,ind,axis=0)
    P=label.sum()
    N=len(score)-P
    c=np.zeros((len(thresholds)+1,4))
    c[0]=[P,N,0,0]
    for i,th in enumerate(thresholds):
        pos=np.searchsorted(score,th)
        fn=np.sum(label[:pos])
        tn=pos-fn
        tp=P-fn
        fp=N-tn
        c[i+1]=[tp,fp,tn,fn]
    return c

def confusion_matrix(scores_labels: ndarray):
    return _confusion_matrix(scores_labels[:, 0], scores_labels[:, 1])


def _confusion_matrix(scores: ndarray, labels: ndarray):
    indexs = argsort(scores)
    labels = labels[indexs]
    P = labels.sum()
    N = len(labels)-P
    TP, FP, TN, FN = 0, 1, 2, 3
    vec = [P, N, 0, 0]
    confusions = zeros((len(labels),4))
    for i,label in enumerate(labels):
        if label == 1:
            vec[FN] += 1
            vec[TP] -= 1
        else:
            vec[TN] += 1
            vec[FP] -= 1
        confusions[i]=vec
    return confusions


def P_R_TPR_FPR(confusion_matrix: ndarray):
    TP, FP, TN, FN = 0, 1, 2, 3
    TP = confusion_matrix[:, TP]
    FP = confusion_matrix[:, FP]
    TN = confusion_matrix[:, TN]
    FN = confusion_matrix[:, FN]
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    FPR = FP/(TN+FP)
    TPR = R
    return stack((P, R, TPR, FPR))


def AUC(X: ndarray, Y: ndarray):
    idx=np.argsort(X)
    x_,y_=0.,0.
    s=0.
    for i in idx:
        x,y=X[i],Y[i]
        s=s+(y+y_)*(x-x_)/2
        x_,y_=x,y
    return s


def macro_P_R_TPR_FPR(confusion_matrixs: ndarray):
    p_r_tpr_fpr = stack(list(map(P_R_TPR_FPR, confusion_matrixs)))
    p_r_tpr_fpr = average(p_r_tpr_fpr, axis=0)
    return p_r_tpr_fpr


def micro_P_R_TPR_FPR(confusion_matrixs: ndarray):
    confusion_matrix = sum(confusion_matrixs, axis=0)
    return P_R_TPR_FPR(confusion_matrix)


def draw_PR_ROC(p: ndarray, r: ndarray, tpr: ndarray, fpr: ndarray,fig_name:str="pr-roc.jpg",save_fig:bool=True):
    methods=["GFPred","CBPred","SCMFDD","LRSSL","MBiRW","HGBI"]
    fig, (ax_l, ax_r) = plt.subplots(nrows=1, ncols=2,figsize=(10, 5))
    # fig.suptitle('Compare')
    ax_l.set_title("(A) ROC curves")
    ax_r.set_title("(B) PR curves")
    ax_r.plot(r, p, label='%s(%.3f)' % ("MTRD",AUC(r,p)))
    ax_l.plot(fpr, tpr, label='%s(%.3f)' % ("MTRD",auc(fpr, tpr)))
    for method in methods:
        tpr=get_result(method,"TPR")
        fpr=get_result(method,"FPR")
        roc=auc(fpr,tpr)
        ax_l.plot(fpr,tpr,label="%s(%.3f)"%(method,roc))
        r=get_result(method,"R")
        p=get_result(method,"P")
        aupr=auc(r,p)+r[0]*p[0]
        # aupr=auc(r,p)
        ax_r.plot(r,p,label="%s(%.3f)"%(method,aupr))
    ax_r.set_xlim(-0.02,1)
    ax_r.set_ylim(-0.02,1)
    ax_l.legend()
    ax_r.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_name)
    else:
        plt.show()


def get_auc_aupr(confusion_matrixs: ndarray):
    p_r_tpr_fpr = stack(list(map(P_R_TPR_FPR, confusion_matrixs)))
    auc_aupr=array([[auc(fpr,tpr),auc(r,p)] for (p,r,fpr,tpr) in p_r_tpr_fpr])
    return auc_aupr

def wilcuxon_test(aucs:ndarray,auprs:ndarray,fig_name:str):
    methods=["CBPred","SCMFDD","LRSSL","MBiRW","HGBI"]
    with open(fig_name,"w") as f:
        for method in methods:
            auc_method=get_result(method,'AUC')
            aupr_method=get_result(method,'AUPR')
            # print("P-value(AUC)",method,wilcoxon(auc_method,aucs[:388]))
            # print("P-value(AUPR)",method,wilcoxon(aupr_method,auprs[:388]))
            print("P-value(AUC)",method,wilcoxon(auc_method,[aucs.mean()]*388))
            print("P-value(AUPR)",method,wilcoxon(aupr_method,[auprs.mean()]*388))
            # f.write("P-value(AUC)"+method+ranksums(auc_method,aucs))
            # f.write("P-value(AUPR)"+method+ranksums(aupr_method,auprs))
    print("P-value(AUC)","GFPred",wilcoxon([0.944815302824281]*388,aucs[:388]))
    print("P-value(AUPR)","GFPred",wilcoxon([0.24334134820106645]*388,auprs[:388]))


def draw_top_k(recall_file:str,fig_name:str="top_k.pdf",save_fig:bool=True):
    methods=["GFPred","CBPred","SCMFDD","LRSSL","MBiRW","HGBI"]
    for m in methods:
        r=get_result(m,"TPR")
        for k in [30,60,90,120,150,180,210,240]:
            print(m,":",r[k-1])

    labels = ["Top%d"%k for k in range(30,240+1,30)]
    x = np.arange(len(labels))  # the label locations
    width = 0.13  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 4))

    recalls=[]
    r=np.loadtxt(recall_file)
    r.sort()
    for k in range(30,240+1,30):
        recalls.append(r[k-1])
    ax.bar(x - 2.5*width, recalls, width, label="MTRD")

    i=1
    for m in methods:
        recalls=[]
        r=get_result(m,"TPR")
        for k in range(30,240+1,30):
            recalls.append(r[k-1])
        ax.bar(x - 2.5*width+width*i, recalls, width, label=m)
        i+=1

    ax.set_ylabel('Recall')
    # ax.set_title('Recalls by Top K')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(top=1.15)
    ax.legend(loc="upper center",ncol=7)
    fig.tight_layout()
    if save_fig:
        plt.savefig(fig_name)
    else:
        plt.show()
