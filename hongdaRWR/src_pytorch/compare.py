from matplotlib import pyplot as plt
from numpy import loadtxt
from os.path import join,realpath,split
from sklearn.metrics import auc
from scipy.stats import ranksums

dir_data=join(split(realpath(__file__))[0],"../data/all_result")

def get_result(method:str,metrics:str):
    method_table={
        "CBPred":1,
        "LRSSL":2,
        "SCMFDD":3,
        "HGBI":4,
        "MBiRW":5,
        "GFPred":6
    }
    assert method in ["CBPred","SCMFDD","LRSSL","MBiRW","HGBI","GFPred"],'method must in ["CBPred","SCMFDD","LRSSL","MBiRW","HGBI","GPred"]'
    metrics=metrics.upper()
    assert metrics in ["P","R","TPR","FPR","AUC","AUPR"],'metrics must in ["P","R","TPR","FPR","AUC","AUPR"]'
    id_method=method_table[method]
    if method=="GFPred": 
        if metrics in ["P","R"]:
            metrics="TPR" if metrics=="R" else metrics
            return loadtxt(join(dir_data,"gao","1","%dmean5_%s.txt"%(0,metrics)))
        else:
            return loadtxt(join(dir_data,"gao","2","%dmean5_%s.txt"%(0,metrics)))
    elif method=="CBPred":
        if metrics == "P":
            return loadtxt(join(dir_data,"1mean5_P.txt"))
        elif metrics == "R":
            return loadtxt(join(dir_data,"1mean5_TPR.txt"))
        elif metrics == "TPR":
            return loadtxt(join(dir_data,"11mean5_TPR.txt"))
        elif metrics == "FPR":
            return loadtxt(join(dir_data,"11mean5_FPR.txt"))
    metrics="TPR" if metrics=="R" else metrics
    if metrics in ["AUPR","AUC"]:
        return loadtxt(join(dir_data,"%d%s.txt"%(id_method,metrics.lower())))
    return loadtxt(join(dir_data,"%dmean5_%s.txt"%(id_method,metrics)))

if __name__ == "__main__":
    methods=["CBPred","SCMFDD","LRSSL","MBiRW","HGBI"]
    fig, (ax_l, ax_r) = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))
    fig.suptitle('Compare')
    ax_l.set_title("(a)ROC curves")
    ax_r.set_title("(b)PR curves")
    for method in methods:
        tpr=get_result(method,"TPR")
        fpr=get_result(method,"FPR")
        roc=auc(fpr,tpr)
        print(tpr.shape)
        ax_l.plot(fpr,tpr,label="%s(%.3f)"%(method,roc))
        r=get_result(method,"R")
        p=get_result(method,"P")
        aupr=auc(r,p)+r[0]*p[0]
        ax_r.plot(r,p,label="%s(%.3f)"%(method,aupr))
    ax_l.legend()
    ax_r.legend()
    plt.savefig("compare.jpg",dip=600)
        
        
