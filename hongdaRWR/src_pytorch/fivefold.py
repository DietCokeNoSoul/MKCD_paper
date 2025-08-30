from math import inf
from os import makedirs
from os.path import exists, join

import numpy as np
from matplotlib import pyplot as plt
from numpy import argsort, concatenate, empty, ndarray, stack
from sklearn.metrics import auc, precision_score, recall_score, roc_auc_score
from torch import cat, device, load, save
from torch.autograd import no_grad
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import FusionDataset, MultiDrugDisease, RandomDrugDisease
from model import MyModel
from preprocess import (Cosine_similarity, associations_drug_disease,
                        get_diseases, get_drugs, similarity_disease,
                        similarity_drug)
from utils import (folds_of_cross_validation, get_positives_negatives,
                   load_cross_test, load_dataset, load_labels, load_model,
                   load_scores, reset_associations, save_cross_validation,
                   save_dataset, save_labels, save_model, save_scores)
from visualization import draw_PR_ROC, draw_top_k, wilcuxon_test,confusion,confusion_by_threshold
from wellCharacterizedDrugs import (folds_average_well_charaterized,
                                      get_folds_drugs_auc_aupr,
                                      get_well_characterized_drugs,
                                      to_folds_data)


def init_cross_validation(n_folds:int,RRs,DDs,RD,steps,alpha):
    positives,negatives=get_positives_negatives(RD)
    negatives=negatives[:len(positives)]
    folds_pos=folds_of_cross_validation(positives,n_folds)
    folds_neg=folds_of_cross_validation(negatives,n_folds)
    for fold in range(n_folds):

        pos=folds_pos[fold]
        neg=folds_neg[fold]
        rd=reset_associations(RD,pos["test"])
        rdr=Cosine_similarity(rd)
        trains=concatenate([pos["train"],neg["train"]])
        validates=concatenate([pos["test"],neg["test"]])
        save_cross_validation(fold,trains,validates,RD)

        train_left=RandomDrugDisease(trains,alpha,steps,rd,RRs+[rdr],DDs,rd)
        train_right=MultiDrugDisease(trains,rd,RRs+[rdr],DDs,rd)
        dataset=FusionDataset(train_left,train_right)
        save_dataset(fold,dataset,"trainset")
        test_left=RandomDrugDisease(validates,alpha,steps,rd,RRs+[rdr],DDs,RD)
        test_right=MultiDrugDisease(validates,rd,RRs+[rdr],DDs,RD)
        dataset=FusionDataset(test_left,test_right)
        save_dataset(fold,dataset,"testset")

def find_lr(init_lr,max_lr,beta):
    pass

def fold_prepare(fold:int,lr_init,config_left,config_right,cuda):
    trains=load_dataset(fold,"trainset")
    trains.to_device(cuda)
    tests=load_dataset(fold,"testset")
    tests.to_device(cuda)
    model=MyModel(config_left,config_right)
    model=model.to(cuda)
    optim=Adam(model.parameters(),lr_init)
    scheduler=ReduceLROnPlateau(optim,factor=0.3,patience=3)
    return model,optim,scheduler,trains,tests,

def fold_train(fold,lr_init:float,epochs:int,config_left,config_right,criterion,cuda):
    model,optim,scheduler,trains,validates=fold_prepare(fold,lr_init,config_left,config_right,cuda)
    train_loader=DataLoader(trains,batch_size=32,shuffle=True)
    validate_loader=DataLoader(validates,batch_size=32,shuffle=False)
    loss_best=inf
    loss_average=0.
    beta=0.9
    for epoch in range(epochs):
        loss_validate=0.
        loss_train=0.

        for data_left,data_right,label in train_loader:
            data_left = data_left.float()
            data_right = data_right.float()
            target = label.float()

            o_cat, o_left, o_right = model(data_left, data_right)

            loss_left = criterion(o_left, target.view_as(o_left))
            loss_right = criterion(o_right, target.view_as(o_right))
            loss_cat = criterion(o_cat, target.view_as(o_cat))
            lambda_cat,lambda_left,lambda_right=weight
            loss = lambda_cat*loss_cat+lambda_left*loss_left+lambda_right*loss_right

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_train+=loss.item()*o_cat.shape[0]

        
        with no_grad():
            for data_left,data_right,label in validate_loader:
                data_left,data_right=data_left.float(),data_right.float()
                label=label.float()
                o=model.predict(data_left,data_right)
                loss=criterion(o,label.view_as(o))
                loss_validate+=loss.item()*o.shape[0]

        loss_average=loss_average*beta+(1-beta)*(loss_validate/len(validates)+loss_train/len(trains))
        loss_smooth=loss_average/(1-beta**(epoch+1))

        if epoch==1 or loss_smooth<=loss_best:
            print("epoch:%d lr: %f , train loss: %f, valitate loss %f"%(epoch,optim.param_groups[0]["lr"],loss_train,loss_validate))
            save_model(fold,model,"model")
            loss_best=loss_smooth
        scheduler.step(loss_train)
        if optim.param_groups[0]["lr"]<=1e-6 or loss_smooth>4*loss_best:
            break


def fold_validate(fold:int,cuda):
    dataset=load_dataset(fold,"testset")
    dataset.to_device(cuda)
    pairs=load_cross_test(fold)
    dataset.reset_pairs(pairs)
    model=MyModel(config_left,config_right)
    load_model(fold,model,"model")
    model=model.to(cuda)
    loader=DataLoader(dataset,batch_size=64)
    scores=[]
    labels=[]
    with no_grad():
        for data_left,data_right,label in loader:
            data_left,data_right=data_left.float(),data_right.float()
            o=model.predict(data_left,data_right)
            scores.append(o.cpu())
            labels.append(label.cpu())
    labels=concatenate(labels)
    scores=concatenate(scores)
    save_scores(fold,scores)
    save_labels(fold,labels)

    

def fold_confusion(fold:int):
    scores=load_scores(fold)
    labels=load_labels(fold)
    pairs=load_cross_test(fold)
    drugs,diseases,mat_score,mat_label,groups,len_min=group_filter(scores,labels,pairs)
    # 截取有效的疾病数目
    mat_label=mat_label[groups,:len_min]
    mat_score=mat_score[groups,:len_min]
    # 计算混淆矩阵
    result=[]
    for score,label in zip(mat_score,mat_label):
        # c=confusion_by_threshold(score,label,np.linspace(0,1,1000))
        c=confusion(score,label)
        result.append(c)
    result=stack(result)
    result=result.sum(axis=0)
    return result,len_min


def group_filter(scores:ndarray,labels:ndarray,pairs:ndarray,num_positive:int=1,group_by:str="drug"):
    '''
    '''
    drugs=get_drugs()
    diseases=get_diseases()

    mat_score=np.full((len(drugs),len(diseases)),-1.,dtype=float)
    mat_label=np.full((len(drugs),len(diseases)),-1)
    indexs=pairs.transpose()
    i,j=indexs
    mat_score[i,j]=scores.flatten()
    mat_label[i,j]=labels.flatten()
    if group_by !="drug":
        mat_label=mat_label.transpose()
        mat_score=mat_score.transpose()
    # 筛选出有效的分组
    groups=np.where((mat_label>0).sum(axis=1)>num_positive)[0]
    # 获得最少的分组长度
    numbers=(mat_label[groups]>-1).sum(axis=1)
    len_min=numbers.min()
    
    # 根据预测得分，由高到低排序
    ind=argsort(-mat_score,axis=1)
    mat_label=np.take_along_axis(mat_label,ind,axis=1)
    mat_score=np.take_along_axis(mat_score,ind,axis=1)
    
    if group_by=="drug":
        group_name=drugs[groups]
        item_name=diseases[ind]
    else:
        group_name=diseases[groups]
        item_name=drugs[ind]
    return group_name,item_name,mat_score,mat_label,groups,len_min

def draw_figure_0(data_dir:str,fig_name:str):
    cs=[]
    for fold in range(n_folds):
        c,_=fold_confusion(fold)
        print(c.shape)
        cs.append(c)
    lens=[len(c) for c in cs]
    len_min=min(lens)
    cs=[c[:len_min,:] for c in cs]
    cs=stack(cs)
    c=cs.sum(axis=0)
    TP,FP,TN,FN=0,1,2,3
    r=c[:,TP]/(c[:,TP]+c[:,FN])
    p=c[:,TP]/(c[:,TP]+c[:,FP])
    
    fpr=c[:,FP]/(c[:,FP]+c[:,TN])
    tpr=r
    np.savetxt(join(data_dir,"r.txt"),r)
    np.savetxt(join(data_dir,"p.txt"),p)
    np.savetxt(join(data_dir,"tpr.txt"),tpr)
    np.savetxt(join(data_dir,"fpr.txt"),fpr)
    
    draw_PR_ROC(p,r,tpr,fpr,fig_name)
    

def draw_figure_1(data_dir:str,fig_name:str):
    mat_labels=[]
    mat_scores=[]
    groups=[]
    lmi=16000
    for fold in range(n_folds):
        scores=load_scores(fold)
        labels=load_labels(fold)
        pairs=load_cross_test(fold)
        drug,disease,mat_score,mat_label,group,len_min=group_filter(scores,labels,pairs)
        mat_labels.append(mat_label)
        mat_scores.append(mat_score)
        groups.append(group)
        lmi=min(lmi,len_min)
    mat_labels=[mat_labels[i][groups[i],:lmi] for i in range(n_folds)]
    mat_scores=[mat_scores[i][groups[i],:lmi] for i in range(n_folds)]
    mat_scores=concatenate(mat_scores)
    mat_labels=concatenate(mat_labels)
    label=mat_labels[mat_labels>-1]
    score=mat_scores[mat_labels>-1]
    
    # c=confusion_by_threshold(score,label,np.linspace(0,1,1000))
    c=confusion(score,label)
    TP,FP,TN,FN=0,1,2,3
    r=c[:,TP]/(c[:,TP]+c[:,FN])
    p=c[:,TP]/(c[:,TP]+c[:,FP])
    fpr=c[:,FP]/(c[:,FP]+c[:,TN])
    tpr=r
    np.savetxt(join(data_dir,"r.txt"),r)
    np.savetxt(join(data_dir,"p.txt"),p)
    np.savetxt(join(data_dir,"tpr.txt"),tpr)
    np.savetxt(join(data_dir,"fpr.txt"),fpr)
    
    draw_PR_ROC(p,r,tpr,fpr,fig_name)


if __name__ == "__main__":
    
    epochs=50
    methods = [
        # "jaccard",
        # "SMC",
        "cosine"
    ]
    features = [
        "disease",
        "pubchem",
        "target_domain",
        "target_go"
    ]
    RD=associations_drug_disease()
    RRs=[similarity_drug(feature,method) for method in methods for feature in features]
    DDs=[similarity_disease()]

    n_folds=5
    steps=6
    alpha=0.9
    config_conv= [16, 'M', 32, 'M']
    config_right=[(len(RRs)+1)*len(DDs), sum(RD.shape), config_conv]
    config_left=[2*sum(RD.shape),80,(len(RRs)+1)*len(DDs),steps]
    weight=[.1,.1,.8]
    lr=0.001

    positives,negatives=get_positives_negatives(RD)
    negatives=negatives[:len(positives)]
    folds_pos=folds_of_cross_validation(positives,n_folds)
    folds_neg=folds_of_cross_validation(negatives,n_folds)

    cuda=device("cuda:0")
    criterion=BCELoss()

    for try_id in range(50):
        result_dir="./results/%d/"%try_id

        if not exists(result_dir):
            makedirs(result_dir)

        print("初始化交叉验证分组")
        init_cross_validation(n_folds,RRs,DDs,RD,steps,alpha)

        print("开始训练")
        for fold in range(n_folds):
            print("当前fold ",fold)
            fold_train(fold,lr,epochs,config_left,config_right,criterion,cuda)

        print("开始验证")
        for fold in range(n_folds):
            print("当前fold ",fold)
            fold_validate(fold,cuda)

        print("绘制图表")
        draw_figure_0(result_dir,join(result_dir,"roc.jpg"))
        # draw_figure_1(result_dir,join(result_dir,"roc.jpg"))

        print("药物预测分析及特征明显药物数据统计")
        to_folds_data(result_dir)
        get_well_characterized_drugs(result_dir)
        folds_average_well_charaterized(result_dir)
        get_folds_drugs_auc_aupr(result_dir)
        
        print("Top K recall")
        draw_top_k(join(result_dir,"r.txt"),join(result_dir,"topk.jpg"))




