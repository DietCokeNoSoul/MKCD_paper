import os
import shutil
from typing import *

import numpy as np
import pandas as pds
from numpy.core.numeric import ones
from sklearn.metrics import (auc, precision_recall_curve, precision_score,
                             recall_score, roc_curve)
from torch import no_grad
from torch.nn import BCELoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from dataloader import RandomDrugDisease
from model import MyModel
from preprocess import (associations_drug_disease, similarity_disease,
                        similarity_drug,get_drugs,get_diseases)
from visualization import P_R_TPR_FPR, confusion, draw_PR_ROC, draw_top_k,wilcuxon_test


def fold_path(fold: int, dir: str = "runs") -> str:
    return os.path.join(dir, "fold_%d" % (fold+1))


def init_cross_validate(folds: int, magnification: int = 1, dir: str = "runs") -> NoReturn:
    RD = associations_drug_disease()
    positives = np.array(np.where(RD)).T
    negatives = np.array(np.where(RD == 0)).T
    np.random.shuffle(negatives)
    negatives = negatives[:len(positives)*magnification]
    print("正样本数量：", len(positives))
    print("负样本数量：", len(negatives))
    each_group = len(positives)//folds
    lens = [each_group]*(folds-1)
    lens.append(len(positives)-sum(lens))
    print("五倍交叉划分数量", lens)
    if os.path.exists(dir):
        print("目录‘", dir, "’已存在,并且将被覆盖")
        shutil.rmtree(dir)
    for i, (p, n) in enumerate(zip(random_split(positives, lens),random_split(negatives, lens))):
        print("Fold %d" % (i+1), ":")
        p_i = np.array(list(p))
        n_i = np.array(list(n))
        print("\t当前测试集正样本数量=%d" % len(p_i))
        rd = RD.copy()
        rd[p_i[:, 0], p_i[:, 1]] = 0
        mask = ones(RD.shape, bool)
        mask[negatives[:, 0], negatives[:, 1]] = False
        mask[positives[:, 0], positives[:, 1]] = False
        mask[p_i[:, 0], p_i[:, 1]] = True
        mask[n_i[:, 0], n_i[:, 1]] = True
        save_path = fold_path(i, dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, "test_mask"), mask)
        np.save(os.path.join(save_path, "train_rd"), rd)


def train_data():
    methods = [
        "cosine"
    ]
    features = [
        "disease",      # 不使用疾病计算相似性
        "pubchem",
        "target_domain",
        "target_go"
    ]
    RD = associations_drug_disease()
    RRs = [similarity_drug(f, m) for f in features for m in methods]
    DDs = [similarity_disease()]
    return RD, RRs, DDs


def find_param_cross_validate(folds: int, dir: str = "runs", epochs: int = 30, lr: float = 0.001, steps: int = 4, alpha: float = 0.3, config_conv=[16, 'M', 32, 'M'],hidden:int=80) -> NoReturn:
    RD, RRs, DDs = train_data()
    config_left = [2*sum(RD.shape), hidden, len(RRs)*len(DDs), steps]
    config_right = [len(RRs)*len(DDs), sum(RD.shape), config_conv]
    for fold in range(folds):
        writer=SummaryWriter(os.path.join(dir,"board"))
        save_path = fold_path(fold)
        rd = np.load(os.path.join(save_path, "train_rd.npy"))
        mask = np.load(os.path.join(save_path, "test_mask.npy"))
        samples = np.array(np.where(mask == False)).T
        print("\t训练样本个数%d" % len(samples))
        ds_train = RandomDrugDisease(samples, alpha, steps, rd, RRs, DDs, RD)
        model = MyModel(config_left, config_right).cuda()
        criterion = CrossEntropyLoss()
        optim = Adam(model.parameters(), 0.001)
        train_loss=[]
        test_loss=[]
        TP=np.array(np.where(RD != rd)).T # 测试集中的真实正样本
        TN=np.array(np.where(np.logical_and(mask,RD==False))).T # 测试集中的真实负样本
        for e in trange(epochs, desc="Fold %d" % (fold+1)):
            for r, d, dl, dr, label in DataLoader(ds_train, batch_size=32, shuffle=True):
                o_cat, o_left, o_right = model(
                    dl.float().cuda(), dr.float().cuda())
                # o_cat = model.right_result(dr.float().cuda())
                label = label.long().cuda()
                loss_left = criterion(o_left, label)
                loss_right = criterion(o_right, label)
                loss_cat = criterion(o_cat, label)
                lambda_cat, lambda_left, lambda_right = (0.6, 0.2, 0.2)
                loss = lambda_cat*loss_cat+lambda_left*loss_left+lambda_right*loss_right
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss.append(loss_cat.item())
            np.random.shuffle(TN)
            samples_test=np.concatenate((TP,TN[:7*len(TP)]),axis=0)
            ds_test = RandomDrugDisease(samples_test, alpha, steps, rd, RRs, DDs, RD)
            with no_grad():
                for r, d, dl, dr, label in DataLoader(ds_test, batch_size=3051, shuffle=False):
                    # o_cat, o_left, o_right = model(dl.float().cuda(), dr.float().cuda())
                    o_left = model.model_left.get_result(dl.float().cuda())
                    loss=criterion(o_cat,label.long().cuda())
                    test_loss.append(loss.item())
            writer.add_scalars("loss",{'train%d'%(fold+1):np.array(train_loss).mean(),'test%d'%(fold+1):np.array(test_loss).mean()},e)
        writer.close()


def train_each_cross_validate(folds: int, dir: str = "runs", epochs: int = 30, lr: float = 0.001, steps: int = 4, alpha: float = 0.3, config_conv=[16, 'M', 32, 'M'],hidden:int=80) -> NoReturn:
    RD, RRs, DDs = train_data()
    config_left = [2*sum(RD.shape), hidden, len(RRs)*len(DDs), steps]
    config_right = [len(RRs)*len(DDs), sum(RD.shape), config_conv]
    for fold in range(folds):
        save_path = fold_path(fold)
        rd = np.load(os.path.join(save_path, "train_rd.npy"))
        mask = np.load(os.path.join(save_path, "test_mask.npy"))
        samples = np.array(np.where(mask == False)).T
        print("\t训练样本个数%d" % len(samples))
        dataset = RandomDrugDisease(samples, alpha, steps, rd, RRs, DDs, RD)
        model = MyModel(config_left, config_right).cuda()
        criterion = CrossEntropyLoss()
        optim = Adam(model.parameters(), 0.001)
        for e in trange(epochs, desc="Fold %d" % (fold+1)):
            for r, d, dl, dr, label in DataLoader(dataset, batch_size=32, shuffle=True):
                o_cat, o_left, o_right = model(
                    dl.float().cuda(), dr.float().cuda())
                # o_cat = model.right_result(dr.float().cuda())
                label = label.long().cuda()
                loss_left = criterion(o_left, label)
                loss_right = criterion(o_right, label)
                loss_cat = criterion(o_cat, label)
                lambda_cat, lambda_left, lambda_right = (0.6, 0.2, 0.2)
                loss = lambda_cat*loss_cat+lambda_left*loss_left+lambda_right*loss_right
                # loss=loss_cat
                optim.zero_grad()
                loss.backward()
                optim.step()
        tests = np.stack(np.where(mask)).T
        scores = np.zeros(mask.shape)
        predicts=np.zeros(mask.shape,int)
        with no_grad():
            for r, d, dl, dr, label in DataLoader(RandomDrugDisease(tests, alpha, steps, rd, RRs, DDs, RD), batch_size=3072):
                y = model.predict(dl.float().cuda(), dr.float().cuda())
                # y = model.right_result(dr.float().cuda())
                y_= y.cpu()
                scores[r, d] = y_[:, 1].reshape(r.numel())
                predicts[r,d] = y_.argmax(dim=1).reshape(r.numel())
        save_path = fold_path(fold, dir)
        np.save(os.path.join(save_path, "test_score"), scores)
        np.save(os.path.join(save_path, "score_mask"), mask)
        np.save(os.path.join(save_path,"test_class"), predicts)


def combine_all_cross_validations(folds, dir: str = "runs", threshold: int = 2):
    RD = associations_drug_disease()
    scores = np.zeros(RD.shape, dtype=float)
    cnt = np.zeros(RD.shape, dtype=int)
    print("\t测试分组过滤阈值=%d" % threshold)
    label_zero=np.zeros(RD.shape,dtype=int)
    label_ones=np.zeros(RD.shape,dtype=int)
    for fold in range(folds):
        save_path = fold_path(fold)
        s = np.load(os.path.join(save_path, "test_score.npy"))
        c = np.load(os.path.join(save_path, "score_mask.npy"))
        l = np.load(os.path.join(save_path, "test_class.npy"))
        label_ones[c]=label_ones[c]+(l[c]==1)
        label_zero[c]=label_zero[c]+(l[c]==0)
        scores = scores+s
        cnt = cnt+c
    scores = scores/np.where(cnt > 0, cnt, 1)
    label  = np.where(label_ones>label_zero,1,0)
    mask = cnt > 0
    useful = np.logical_and(RD == 1, mask).sum(axis=1, keepdims=True) >= threshold
    print("\t可用测试药物个数：%d" % useful.sum())
    mask_used = np.logical_and(mask, useful)
    print("\t测试样本个数：%d" % mask_used.sum())
    np.save(os.path.join(dir, "scores"), scores)
    np.save(os.path.join(dir, "mask"), mask)
    np.save(os.path.join(dir, "labels"),label)
    recalls=[]
    mi=mask.shape[1]
    for i in range(mask.shape[0]):
        if useful[i]==False: continue
        m=mask[i,:]
        s=scores[i,:]
        l=RD[i,:]
        p,r,_,_=P_R_TPR_FPR(confusion(s[m],l[m]))
        mi=min(mi,len(r))
        recalls.append(r)
    recalls=list(map(lambda x:x[:mi],recalls))
    recalls=np.array(recalls)
    recalls=np.nanmean(recalls,axis=0)
    np.savetxt(os.path.join(dir,"recall.txt"),recalls)




def draw_all_plots(dir: str = "runs",average_by_drugs=True,save_fig=False,threshold=1):
    RD = associations_drug_disease()
    mask = np.load(os.path.join(dir, "mask.npy"))
    predict = np.load(os.path.join(dir, "scores.npy"))
    # idx=np.argsort(-predict,axis=1)
    # mask=np.take_along_axis(mask,idx,axis=1)
    # predict=np.take_along_axis(predict,idx,axis=1)
    # RD=np.take_along_axis(RD,idx,axis=1)
    # print(predict)
    def shortmean(a:list,mi:int):
        a=list(map(lambda x:x[:mi],a))
        a=np.array(a)
        a=np.nanmean(a,axis=0)
        return a
    
    if not average_by_drugs:
        s = predict[mask]
        l = RD[mask]
        p, r, _ = precision_recall_curve(l, s)
        fpr, tpr, _ = roc_curve(l, s)
    else:
        useful = mask.sum(axis=1, keepdims=True)>threshold
        ps=[]
        rs=[]
        fprs=[]
        tprs=[]
        mi=mask.shape[1]
        for i in range(mask.shape[0]):
            if useful[i]==False: continue
            s=predict[i]
            l=RD[i]
            # s[l==1]=np.where(s[l==1]>1-s[l==1],s[l==1],1-s[l==1])
            p,r,tpr,fpr=P_R_TPR_FPR(confusion(s,l))
            mi=min(mi,len(r))
            ps.append(p)
            rs.append(r)
            fprs.append(fpr)
            tprs.append(tpr)
        p,r,fpr,tpr=map(lambda x:shortmean(x,mi),[ps,rs,fprs,tprs])
        fpr,tpr, _ = roc_curve(RD[mask], predict[mask])
    file_name=os.path.join(dir,"PR_ROC.pdf")
    draw_PR_ROC(p[::-1], r[::-1], tpr, fpr, file_name,save_fig)
    
def p_value(dir:str="runs"):
    RD = associations_drug_disease()
    mask = np.load(os.path.join(dir, "mask.npy"))
    predict = np.load(os.path.join(dir, "scores.npy"))
    useful = mask.sum(axis=1, keepdims=True)>=3
    aurocs=[]
    auprs=[]
    for i in range(mask.shape[0]):
        if useful[i]==False: continue
        s=predict[i]
        l=RD[i]
        fpr,tpr,_=roc_curve(l,s)
        p,r,_=precision_recall_curve(l,s)
        auroc=auc(fpr,tpr)
        aupr=auc(r,p)
        aurocs.append(auroc)
        auprs.append(aupr)
    
    wilcuxon_test(np.array(auprs)[:388],np.array(auprs)[:388],os.path.join(dir,"p_values.txt"))


def supplementary_list(dir:str="runs",file_name:str="ST.xlsx",topk:int=30,threshold:int=15,best_first:bool=False):
    drugs=get_drugs()
    diseases=get_diseases()
    RD = associations_drug_disease()
    mask = np.load(os.path.join(dir, "mask.npy"))
    mask = np.logical_and(mask,RD==0)
    predict = np.load(os.path.join(dir, "scores.npy"))
    writer=pds.ExcelWriter(os.path.join(dir,file_name),engine="openpyxl")
    ST=[]
    SR=[]
    for i,(p,m) in enumerate(zip(predict,mask)):
        if RD[i].sum() < threshold : continue
        scores=p[m]
        p,r,_=precision_recall_curve(RD[i],p)
        SR.append(auc(r,p))
        names=diseases[m]
        idx=np.argsort(scores)[::-1]
        ds=pds.Series(names[idx][:topk])
        ss=pds.Series(scores[idx][:topk])
        rs=pds.Series([drugs[i]]*topk)
        df=pds.DataFrame({"Drug Name":rs,"Disease Name":ds,"Score":ss})
        ST.append(df)
    if best_first:
        idx=np.argsort(SR)[::-1]
        ST=[ST[i] for i in idx]
    pds.concat(ST,axis=0).to_excel(writer)
    writer.save()