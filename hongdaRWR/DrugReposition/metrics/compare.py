import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pds

from .metrics import Metrics

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, "..", "data", "all_result")
data_dir = os.path.abspath(data_dir)


class OtherMethods(object):

    @staticmethod
    def legal(name: str) -> bool:
        '''
        该对比方法是否存在（如下文件是否存在）。\n

        `data/all_result/{name}/PR_curve/P.txt`\n
        `data/all_result/{name}/PR_curve/R.txt`\n
        `data/all_result/{name}/ROC_curve/TPR.txt`\n
        `data/all_result/{name}/ROC_curve/FPR.txt`\n
        `data/all_result/{name}/AUC.txt`\n
        `data/all_result/{name}/AUPR.txt`\n
        '''
        p = os.path.join(data_dir, name)
        if not os.path.isdir(p):
            return False
        for x in ["TPR", "FPR"]:
            if not os.path.isfile(os.path.join(p, "ROC_curve", "%s.txt" % x)):
                return False
        for x in ["P", "R"]:
            if not os.path.isfile(os.path.join(p, "PR_curve", "%s.txt" % x)):
                return False
        # for x in ["AUC", "AUPR"]:
        #     if not os.path.isfile(os.path.join(p, "%s.txt" % x)):
        #         return False
        return True

    @staticmethod
    def methods() -> Set[str]:
        '''
        可用于比较的其他方法名字。
        '''
        methods = os.listdir(data_dir)
        methods = set(filter(OtherMethods.legal, methods))
        return methods

    @staticmethod
    def metrics(name: str, m: str) -> np.ndarray:
        '''
        方法名为`name`的指标`m`。
        '''
        methods = OtherMethods.methods()
        m = m.upper()
        assert name in methods, "%s is not exists." % name
        assert m in ["FPR", "TPR", "P", "R",
                     "AUC", "AUPR"], "%s is not exists." % m
        if m in ["FPR", "TPR"]:
            name = os.path.join(name, "ROC_curve")
        elif m in ["P", "R"]:
            name = os.path.join(name, "PR_curve")
        file = os.path.join(data_dir, name, "%s.txt" % m)
        return np.loadtxt(file)

    @staticmethod
    def compare(name: str, r: np.ndarray, p: np.ndarray, fpr: np.ndarray, tpr: np.ndarray, save_fig: Optional[str] = None):
        '''
        基于ROC和P-R curve与其它方法对比。
        '''
        methods = OtherMethods.methods()
        plt.rc('font', family='Times New Roman')
        fig, (ax_l, ax_r) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax_r.plot(r, p, label='%s(%.3f)' % (name, Metrics.auc(r, p)))
        ax_l.plot(fpr, tpr, label='%s(%.3f)' % (name, Metrics.auc(fpr, tpr)))
        ax_l.set_title("(A) ROC curves")
        ax_r.set_title("(B) PR curves")
        datas = []
        for m in methods:
            r = OtherMethods.metrics(m, "R")
            p = OtherMethods.metrics(m, "P")
            fpr = OtherMethods.metrics(m, "FPR")
            tpr = OtherMethods.metrics(m, "TPR")
            auc = Metrics.auc(fpr, tpr)
            aupr = Metrics.auc(r, p)
            datas.append({
                "name": m,
                "R": r,
                "P": p,
                "FPR": fpr,
                "TPR": tpr,
                "AUC": auc,
                "AUPR": aupr
            })
        datas.sort(key=lambda x: x["AUPR"], reverse=True)
        for each in datas:
            m = each["name"]
            r, p = each["R"], each["P"]
            fpr, tpr = each["FPR"], each["TPR"]
            auc, aupr = each["AUC"], each["AUPR"]
            ax_r.plot(r, p, label='%s(%.3f)' % (m, aupr))
            ax_l.plot(fpr, tpr, label='%s(%.3f)' % (m, auc))
        ax_l.legend()
        ax_r.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        if save_fig is None:
            plt.show()
        else:
            print("Saved in %s" % save_fig)
            plt.savefig(save_fig)

    @staticmethod
    def topK_recall(name: str, r: np.ndarray, save_fig: Optional[str] = None):
        '''
        与其它方法在top 240下的recall对比。
        '''
        methods = list(OtherMethods.methods())
        plt.rc('font', family='Times New Roman')
        labels = ["Top%d" % k for k in range(30, 240+1, 30)]
        x = np.arange(len(labels))  # the label locations
        width = 0.13  # the width of the bars
        fig, ax = plt.subplots(figsize=(10, 4))
        df={}
        recalls = []
        r=r.copy()
        r.sort()
        for k in range(30, 240+1, 30):
            recalls.append(r[k-1])
        ax.bar(x - 2.5*width, recalls, width, label=name)
        datas = {}
        for m in methods:
            r = OtherMethods.metrics(m, "R")
            p = OtherMethods.metrics(m, "P")
            aupr = Metrics.auc(r, p)
            datas[m] = aupr
        methods.sort(key=lambda x: datas[x], reverse=True)
        df[name]=pds.Series(recalls,labels,name=name)

        i = 1
        for m in methods:
            recalls = []
            r = OtherMethods.metrics(m, "TPR")
            for k in range(30, 240+1, 30):
                recalls.append(r[k-1])
            ax.bar(x - 2.5*width+width*i, recalls, width, label=m)
            df[m]=pds.Series(recalls,labels,name=m)
            i += 1

        ax.set_ylabel('Recall')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(top=1.15)
        ax.legend(loc="upper center", ncol=7)
        fig.tight_layout()
        if save_fig:
            print("Saved in %s" % save_fig)
            plt.savefig(save_fig)
        else:
            plt.show()
        return pds.DataFrame(df)
