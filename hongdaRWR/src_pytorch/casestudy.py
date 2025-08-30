from torch import save,load
import numpy as np
from os.path import join
import csv
from preprocess import get_drugs,get_diseases,associations_drug_disease

def predict(data_dir:str,rd):
    st=load(join(data_dir,"5-folds.data"))
    drugs={}
    for fold in range(5):
        for i in range(763):
            drugs[i]={}
            scores=st[fold][i]["scores"]
            diseases=st[fold][i]["diseases"]
            for s,d in zip(scores,diseases):
                if rd[i,d] == 1: continue
                if d in drugs[i]:
                    drugs[i][d]["n"]+=1
                    drugs[i][d]["score"].append(s)
                else:
                    drugs[i][d]={
                        "n":1,
                        "score":[s]
                    }

    save(drugs,join(data_dir,"predict.data"))

def candidate(data_dir:str):
    drugs=load(join(data_dir,"predict.data"))
    candidates={}
    name_drug=get_drugs()
    name_dis=get_diseases()
    for i in drugs:
        r_name=name_drug[i]
        candidates[r_name]={
            "scores":[],
            "diseases":[]
        }
        for dis in drugs[i]:
            s=sum(drugs[i][dis]["score"])/drugs[i][dis]["n"]
            d_name=name_dis[dis]
            candidates[r_name]["scores"].append(s)
            candidates[r_name]["diseases"].append(d_name)
    for i in candidates:
        scores=np.array(candidates[i]["scores"])
        diseases=np.array(candidates[i]["diseases"])
        ind=np.argsort(-scores)
        diseases=np.take(diseases,ind)
        scores=np.take(scores,ind)
        candidates[i]["scores"]=scores
        candidates[i]["diseases"]=diseases

    save(candidates,join(data_dir,"fine_predict.data"))
    for k in candidates:
        print(k)
        r=candidates[k]
        print(r["scores"][:30])
        print(r["diseases"][:30])
    to_csv(data_dir)
    
def to_csv(data_dir:str):
    candidates=load(join(data_dir,"fine_predict.data"))
    with open(join(data_dir,"predict.csv"),"w",newline='') as f:
        writer=csv.writer(f)
        for k in candidates:
            row=[k]
            r=candidates[k]
            row=row+r["diseases"].tolist()
            writer.writerow(row)
    with open(join(data_dir,"predict_score.csv"),"w",newline='') as f:
        writer=csv.writer(f)
        for k in candidates:
            row=[k]
            r=candidates[k]
            row=row+r["diseases"].tolist()
            writer.writerow(row)
        
if __name__ == "__main__":
    # rd=associations_drug_disease()
    # predict("E:\\Projects\\DrugDisease\\results\\0\\",rd)
    candidate("E:\\Projects\\DrugDisease\\results\\0\\")