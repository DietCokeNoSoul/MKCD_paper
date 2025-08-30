from utils import save_scores,load_scores,save_labels,load_labels,load_cross_test,load
from sklearn.metrics import auc
from visualization import confusion
import json
from torch import save,load
from numpy import array
from math import isnan
from os.path import join
from preprocess import get_drugs

well_characterized_drugs=[
    "ampicillin","cefepime","cefoperazone","cefotaxime","cefotetan",
    "cefoxitin","ceftazidime","ceftizoxime","ceftriaxone","ciprofloxacin",
    "doxorubicin","erythromycin","etoposide","hydrocortisone","itraconazole",
    "levofloxacin","moxifloxacin","ofloxacin"
]
well_characterized_drugs_id=[
    32,113,115,116,117,
    118,121,123,124,150,
    232,259,277,346,379,
    401,483,523
]


def to_folds_data(data_dir:str):
    st=dict()
    for fold in range(5):
        st[fold]={}
        labels=load_labels(fold)
        scores=load_scores(fold).reshape(-1)
        pairs=load_cross_test(fold)
        for i in range(763):
            st[fold][i]={
                "labels":[],
                "scores":[],
                "diseases":[]
            }
        for p,l,s in zip(pairs,labels,scores):
            st[fold][p[0]]["labels"].append(l)
            st[fold][p[0]]["scores"].append(s)
            st[fold][p[0]]["diseases"].append(p[1])

    save(st,join(data_dir,"5-folds.data"))


def get_folds_drugs_auc_aupr(data_dir:str):
    st=load(join(data_dir,"5-folds.data"))
    rs=get_drugs()
    metrics={}
    metrics["folds"]={}
    for fold in range(5):
        metrics["folds"][fold]={}
        metrics["folds"][fold]["drugs"]=[]
        for i in range(763):
            score=st[fold][i]["scores"]
            label=st[fold][i]["labels"]
            drug=rs[i]
            score=array(score)
            label=array(label)
            c=confusion(score,label)
            TP,FP,TN,FN=0,1,2,3
            r=c[:,TP]/(c[:,TP]+c[:,FN])
            p=c[:,TP]/(c[:,TP]+c[:,FP])
            fpr=c[:,FP]/(c[:,FP]+c[:,TN])
            tpr=r
            tmp={
                "name":drug,
                "id":i,
                "positives":sum(label),
                "negatives":len(label)-sum(label),
                # "p":p.tolist(),
                # "r":r.tolist(),
                # "tpr":tpr.tolist(),
                # "fpr":fpr.tolist(),
                "auc":auc(fpr,tpr),
                "aupr":auc(r[:-1],p[:-1])
            }
            metrics["folds"][fold]["drugs"].append(tmp)
        
    with open(join(data_dir,"folds_drug_auc_aupr.json"),"w") as f:
        json.dump(metrics,f)

def get_well_characterized_drugs(data_dir:str):
    st=load(join(data_dir,"5-folds.data"))
    metrics={}
    metrics["folds"]={}
    for fold in range(5):
        metrics["folds"][fold]={}
        metrics["folds"][fold]["drugs"]=[]
        for i,drug in zip(well_characterized_drugs_id,well_characterized_drugs):
            score=st[fold][i]["scores"]
            label=st[fold][i]["labels"]
            score=array(score)
            label=array(label)
            c=confusion(score,label)
            TP,FP,TN,FN=0,1,2,3
            r=c[:,TP]/(c[:,TP]+c[:,FN])
            p=c[:,TP]/(c[:,TP]+c[:,FP])
            fpr=c[:,FP]/(c[:,FP]+c[:,TN])
            tpr=r
            tmp={
                "name":drug,
                "id":i,
                "positives":sum(label),
                "negatives":len(label)-sum(label),
                # "p":p.tolist(),
                # "r":r.tolist(),
                # "tpr":tpr.tolist(),
                # "fpr":fpr.tolist(),
                "auc":auc(fpr,tpr),
                "aupr":auc(r[:-1],p[:-1])
            }
            metrics["folds"][fold]["drugs"].append(tmp)
        
    with open(join(data_dir,"well-characterized.json"),"w") as f:
        json.dump(metrics,f)

def get_drug(drugs,drug):
    for each in drugs:
        if each["name"]==drug:
            return each

def folds_average_well_charaterized(data_dir:str):
    with open(join(data_dir,"well-characterized.json"),"r") as f:
        metrics=json.load(f)
    for fold in range(5):
        drugs=metrics["folds"][str(fold)]["drugs"]
        auc=[a["auc"] for a in drugs if not isnan(a["auc"])]
        aupr=[a["aupr"] for a in drugs if not isnan(a["aupr"])]
        print(sum(auc)/len(auc),sum(aupr)/len(aupr))

    data=[]
    for i,drug in zip(well_characterized_drugs_id,well_characterized_drugs):
        tmp={
            "name":drug,
            "id":i,
            "auc":[],
            "aupr":[]
        }
        for fold in range(5):
            drugs=metrics["folds"][str(fold)]["drugs"]
            d=get_drug(drugs,drug)
            if not isnan(d["auc"]):
                tmp["auc"].append(d["auc"])
            if not isnan(d["aupr"]):
                tmp["aupr"].append(d["aupr"])
        data.append(tmp)
    
    with open(join(data_dir,"5-folds_well-characterized.json"),"w") as f:
        json.dump(data,f)