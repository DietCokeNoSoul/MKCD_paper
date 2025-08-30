import os
import re
from io import BytesIO
from json import loads
from os.path import join
from time import sleep

import numpy as np
import pandas as pds
import urllib3
from lxml import etree
from tqdm import tqdm

http = urllib3.PoolManager()


def select_from_CTD(drug: str, disease: str, retry=3):
    quary = {
        "type": "reference",
        "chemqt": "equals",
        "chem": drug,
        "geneqt": "equals",
        "gene": "",
        "diseaseqt": "contains",
        "disease": disease,
        "goqt": "equals",
        "go": "",
        "taxonqt": "equals",
        "taxon": "",
        "partyqt": "equals",
        "party": "",
        "titleAbstract": "",
        "yearFrom": "0",
        "yearThrough": "0",
        "accId": "",
        "perPage": "50",
        "action": "Search"
    }

    for i in range(retry):
        res = http.request(
            "GET", "https://ctdbase.org/query.go",
            fields=quary
        )
        if res.status == 200:
            ans=res.data.decode("utf-8")
            html = etree.HTML(ans)
            if len(html.xpath("/html/body/form/div[1]")) > 0:
                # print("Not Find !")
                return "NO"
            elif len(html.xpath("/html/body/table")) > 0 or len(html.xpath("/html/body/div[6]/table")) > 0:
                # /html/body/div[6]/table
                # print("Find !")
                num=re.findall(r"[0-9]+ results",ans)
                if len(num)>0:
                    return num[0] 
                return "1 results"
        sleep(0.5)
    print("Error !", drug, disease)
    return "Time Out"


def get_drug_Mesh(name:str,retry=3):
    url="https://ctdbase.org/vocAutoComplete.ajx"
    quary={
        "max":10,
        "voc":"chem",
        "q":name
    }
    res = http.request("GET", url, fields=quary)
    for i in range(retry):
        if res.status == 200:
            js=loads(res.data.decode("ascii"))
                # else :break
            if len(js)!=0:
                return js[0]["acc"]
        sleep(0.5)
    raise Exception("Can not transform name (%s) into MeSH." % name)

def get_disease_MeSH(name: str,retry=3):
    # https://ctdbase.org/basicQuery.go?bqCat=disease&bq=Polymyositis
    url="https://ctdbase.org/basicQuery.go"
    quary={
        "bqCat":"disease",
        "bq":name
    }
    res = http.request("GET", url, fields=quary)
    for i in range(retry):
        if res.status == 200:
            ans=re.findall(r"MESH%3[aA][a-zA-Z][0-9]+",res.data.decode("utf-8"))
            if len(ans)>0:return ans[0].replace("%3A",":").replace(r"%3a",":")
            else:break
        sleep(0.5)
    raise Exception("Can not transform name (%s) into MeSH." % name)


def get_csv(mesh: str, tp:str,retry=3) -> pds.DataFrame:
    # https://ctdbase.org/detail.go?acc=MESH%3AD012852&view=chem&6578706f7274=1&type=disease&d-1332398-e=1
    view,tp= ("disease","chem") if tp=="chem" else ("chem","disease")
    if tp=="chem":
        mesh=mesh[5:]
    quary = {
        "acc": mesh,
        "view": view,
        "6578706f7274": "1",
        "type": tp,
        "d-1332398-e": "1"
    }
    print(mesh)
    for i in range(retry):
        res = http.request(
            "GET", "https://ctdbase.org/detail.go", fields=quary)
        if res.status == 200:
            print(res.data.decode("utf-8"))
            if re.match(r".*html.*", res.data.decode("utf-8")):
                break
            print(res.geturl())
            return pds.read_csv(BytesIO(res.data))
        sleep(0.5)
    raise Exception("Can not find items with %s." % mesh)

def try_load(dir:str,table:dict,tp:str,items,mesh_tab,error_list):
    for d in tqdm(items, total=len(items),desc="Loading %s..."%tp):
        if d in error_list:
            continue
        file_name = join(dir,"CTD_%s.csv" % d)
        if os.path.exists(file_name):
            dis_set = pds.read_csv(file_name)
        else:
            if not os.path.exists(dir):
                os.makedirs(dir)
            try:
                mesh = get_disease_MeSH(d) if tp=="disease" else get_drug_Mesh(d)
                dis_set = get_csv(mesh,tp=tp)
            except Exception as e:
                error_list.add(d)
                continue
            mesh_tab[d]=mesh
            dis_set.to_csv(file_name)
        table[d] = dis_set


def quary_from_CTD(dir: str, in_file: str, out_file: str, skip_err=True):
    rd = pds.read_excel(join(dir, in_file))
    diseases = rd["Disease Name"].unique()
    drugs=rd["Drug Name"].unique()
    dis_tab = {}
    drug_tab={}
    if skip_err:
        errors = np.loadtxt(join(dir, "errlist.txt"), str, delimiter='\n')
    else:
        errors = []
    errors = set(errors)
    mesh_tab={}
    try_load(join(dir,"ctd","disease"),dis_tab,"disease",diseases,mesh_tab,errors)
    try_load(join(dir,"ctd","drug"),drug_tab,"chem",drugs,mesh_tab,errors)
    for k,v in mesh_tab:
        print(k,v)
    if len(errors):
        print("Error list:")
        print(errors)
        np.savetxt(join(dir, "errlist.txt"), np.array(list(errors)), "%s")
    bar = tqdm(rd.iterrows(), total=len(rd))
    now = "Starting..."
    dis2drug=[]
    drug2dis=[]
    ref=[]
    for (i, row) in bar:
        r, d = row[1], row[2]
        if r not in mesh_tab.keys():
            mesh_tab[r]=get_drug_Mesh(r)
        if d not in mesh_tab.keys():
            mesh_tab[d]=get_disease_MeSH(d)
        mesh_r=mesh_tab[r][5:]
        mesh_d=mesh_tab[d]
        if r != now:
            bar.set_description_str("Searching %s..." % r)
            now = r
        rds = "NO"
        if d in dis_tab.keys():
            tb = dis_tab[d]
            ans = tb["Chemical ID"].str.upper() == mesh_r.upper()
            if ans.sum()==0:
                rds="NO"
            else:
                evidence=tb.loc[ans]["Direct Evidence"].unique()
                if len(evidence)==1 and not isinstance(evidence[0],str):
                    rds=str(tb.loc[ans]["Reference Count"].sum())
                else:
                    rds="YES"
        else:
            rds = "NO"
        dis2drug.append(rds)
        if r in drug_tab.keys():
            tb = drug_tab[r]
            ans = tb["Disease ID"].str.upper() == mesh_d.upper()
            if ans.sum()==0:
                rds="NO"
            else:
                evidence=tb.loc[ans]["Direct Evidence"].unique()
                if len(evidence)==1 and not isinstance(evidence[0],str):
                    rds=str(tb.loc[ans]["Reference Count"].sum())
                else:
                    rds="NO"
        else:
            rds = "NO"
        drug2dis.append(rds)
        ref.append(select_from_CTD(r,d))

    with pds.ExcelWriter(join(dir, out_file), engine="openpyxl") as writer:
        pds.concat((rd, pds.Series(dis2drug, name="CTD_dis"),pds.Series(drug2dis,name="CTD_drug"),pds.Series(ref,name="CTD_ref")), axis=1).to_excel(writer)
        writer.save()


# 获取疾病的MeSH->疾病相关的药物
# 药物中查找相关的关联，然后记录（关联证据、引用数）


if __name__ == "__main__":
    # ms = get_disease_MeSH("Skin Diseases Bacterial")
    # print(ms)
    # df = get_disease_csv(ms)
    # print(df)
    quary_from_CTD(r"E:\Projects\DrugDisease\python\record\runs0.252","CaseStudy.xlsx", "CTD.xlsx", skip_err=False)
    # mesh=get_disease_MeSH("Bites  Human")
    # print(mesh)
    # print(get_csv(mesh,"disease"))
