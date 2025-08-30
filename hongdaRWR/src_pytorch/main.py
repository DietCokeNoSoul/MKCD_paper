from numpy import average, ndarray, stack, vstack,concatenate,array,unique
from torch import device, mode, no_grad,from_numpy
from torch.nn import CrossEntropyLoss
from torch.nn.modules.module import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from dataloader import FusionDataset, MultiDrugDisease, RandomDrugDisease
from model import MyModel
from preprocess import (associations_drug_disease, similarity_disease,
                        similarity_drug)
from utils import (
    get_positives_negatives, list_sub, load_cross_test, load_cross_train,
    load_dataset, load_labels, load_models, load_scores, partition,
    remaining_pairs, reset_associations, save_cross_validation, save_dataset,
    save_labels, save_models, save_scores, simples_with_positive, take_simple)

from visualization import (confusion_matrix, draw_PR_ROC, macro_P_R_TPR_FPR,
                           micro_P_R_TPR_FPR,wilcuxon_test,get_auc_aupr)


def cross_folds(folds: int, RD: ndarray):
    positives, negatives = get_positives_negatives(RD)
    negatives = negatives[:len(positives)]
    Rs, Ds = RD.shape
    all_pairs = array([(r, d)for r in range(Rs) for d in range(Ds)])
    for fold in range(folds):
        train_pairs = remaining_pairs(positives, negatives, fold, folds)
        test_pairs = list_sub(all_pairs, train_pairs)
        reset_pairs = partition(positives, fold, folds)
        save_cross_validation(fold, train_pairs, test_pairs, reset_pairs)


def train(model:Module,dataset:FusionDataset,fold,epochs):
    cuda=device("cuda:0")
    criterion = CrossEntropyLoss()
    pairs,_=load_cross_train(fold)
    dataset.left.pairs=pairs
    dataset.right.pairs=pairs
    dataset.to_device(cuda)
    model.to(cuda)
    opt=Adam(model.parameters(),lr=0.003)
    model.train()
    dataloader=DataLoader(dataset,batch_size=32)
    for epoch in range(epochs):
        for input_left,input_right,target in dataloader:
            input_left=input_left.float()
            input_right=input_right.float()
            target=target.long()
            o,o_left,o_right=model(input_left,input_right)
            loss=criterion(o,target)+criterion(o_left,target)+criterion(o_right,target)

            opt.zero_grad()
            loss.backward()
            opt.step()


def train_folds(folds: int, epochs:int,RD: ndarray, RRs: list, DDs: list):
    config_conv= [16, 'M', 32, 'M']
    steps=6
    alpha=0.6
    config_left=[len(RRs)*len(DDs), sum(RD.shape), config_conv]
    config_right=[2*sum(RD.shape),200,len(RRs)*len(DDs),steps]
    for fold in range(folds):
        _,reset_rd=load_cross_train(fold)
        rd=reset_associations(RD, reset_rd)
        sets_left=MultiDrugDisease([],rd,RRs,DDs)
        sets_right=RandomDrugDisease([],alpha,steps,rd,RRs,DDs)
        sets=FusionDataset(sets_left,sets_right)
        model=MyModel(config_left,config_right)
        train(model,sets,fold,epochs)
        save_models(fold,[model],["model"])
    pass

def validate(fold,cuda,RD: ndarray, RRs: list, DDs: list):
    steps=6
    alpha=0.6
    models=load_models(fold,["model"])
    pairs=load_cross_test(fold)
    model=models[0]
    _,reset_rd=load_cross_train(fold)
    rd=reset_associations(RD, reset_rd)
    # rd=RD
    left=MultiDrugDisease(pairs,rd,RRs,DDs)
    left.RD=from_numpy(RD.copy())
    right=RandomDrugDisease(pairs,alpha,steps,rd,RRs,DDs)
    right.RD=from_numpy(RD.copy())
    dataset=FusionDataset(left,right)
    dataset.to_device(cuda)
    model.to(cuda)
    model.eval()
    dataloader=DataLoader(dataset,batch_size=2048)
    outputs=[]
    labels=[]
    with no_grad():
        for input_left,input_right,target in dataloader:
            input_left=input_left.float()
            input_right=input_right.float()
            target=target.long()
            o=model.predict(input_left,input_right)
            outputs.append(o.cpu().numpy())
            labels.append(target.cpu().numpy())
    return concatenate(outputs,axis=0),concatenate(labels,axis=0)


def validate_folds(folds:int,RD: ndarray, RRs: list, DDs: list):
    cuda=device("cuda:0")
    for fold in range(folds):
        scores, labels=validate(fold,cuda,RD,RRs,DDs)
        save_scores(fold, scores)
        save_labels(fold, labels)


def filter_effective(pairs, labels, scores,group_by="drug"):
    Rs, Ds = RD.shape
    nums=Rs if group_by=="drug" else Ds
    index=0 if group_by=="drug" else 1
    baskets = [list() for i in range(nums)]
    effective = [False for i in range(nums)]
    for pair, label, score in zip(pairs, labels, scores):
        baskets[pair[index]].append((score, label))
        if label == 1:
            effective[pair[index]] = True
    baskets = list(map(vstack, baskets))
    baskets = [basket for basket, select in zip(baskets, effective) if select]
    count = [basket.shape[0] for basket in baskets]
    shortest = min(count)
    reshaped_baskets = []
    for basket in baskets:
        reshaped_baskets.append(take_simple(basket, shortest))
    reshaped_baskets = stack(reshaped_baskets)
    baskets = simples_with_positive(reshaped_baskets)
    return baskets


def calculate_P_R_TPR_FPR(folds: int,calculate_by="drug",fig_name:str="fig.jpg"):
    p_r_tpr_fpr_list = []
    for fold in range(folds):
        scores = load_scores(fold)
        labels = load_labels(fold)
        test = load_cross_test(fold)
        if calculate_by =="drug":
            drugs = filter_effective(test, labels, scores)
            cms = stack(list(map(confusion_matrix, drugs)))
        else:
            disease = filter_effective(test,labels,scores,"disease")
            cms = stack(list(map(confusion_matrix, disease)))
        # P, R, TPR, FPR = micro_P_R_TPR_FPR(cms)
        P, R, TPR, FPR = macro_P_R_TPR_FPR(cms)
        p_r_tpr_fpr_list.append(stack((P, R, TPR, FPR)))
    count = [each.shape[1] for each in p_r_tpr_fpr_list]
    shortest = min(count)
    resampling_list = []
    for each in p_r_tpr_fpr_list:
        resampling_list.append(take_simple(each.T, shortest).T)
    p, r, tpr, fpr = average(stack(resampling_list), axis=0)
    draw_PR_ROC(p, r, tpr, fpr,fig_name)
    

def calculate_P_values(folds:int,calculate_by="drug"):
    aucs=[]
    for fold in range(folds):
        scores = load_scores(fold)
        labels = load_labels(fold)
        test = load_cross_test(fold)
        if calculate_by =="drug":
            drugs = filter_effective(test, labels, scores)
            cms = stack(list(map(confusion_matrix, drugs)))
        else:
            disease = filter_effective(test,labels,scores,"disease")
            cms = stack(list(map(confusion_matrix, disease)))
        auc_aupr=get_auc_aupr(cms)
        aucs.append(auc_aupr)
    length=min([len(x) for x in aucs])
    # aucs=concatenate(aucs)
    aucs=[x[:length,...] for x in aucs]
    aucs=stack(aucs)
    aucs=average(aucs,axis=0)
    wilcuxon_test(aucs[:,0],aucs[:,1])

if __name__ == "__main__":
    folds=5
    methods = [
        "cosine"
    ]
    features = [
        "disease",
        "pubchem",
        "target_domain",
        "target_go"
    ]
    for epoch in [30,40,50,60,70,80,90,100]:
            RD=associations_drug_disease()
            # RRS=[similarity_drug(f,m) for f in features for m in methods]
            # DDS=[similarity_disease()]
            # print("划分五倍交叉样本")    
            # cross_folds(folds,RD)
            # print("训练")
            # train_folds(folds,epoch,RD,RRS,DDS)
            # print("验证")
            # validate_folds(folds,RD,RRS,DDS)
            print("绘制比较效果图")
            calculate_P_R_TPR_FPR(folds,calculate_by="disease",fig_name="%d.jpg"%epoch)
            print("计算p-value")
            calculate_P_values(folds,calculate_by="disease")
            break