import os
import numpy as np
from typing import *

data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "origin")

def load_csv(file_map: dict, name: str) -> np.ndarray:
    file_name = os.path.join(data_dir, file_map[name])
    data = np.loadtxt(file_name, dtype=str, delimiter=',')
    return data


class PropertyFactory(object):
    def get(*args): ...


class DrugProperty(PropertyFactory):
    '''药物属性工厂，产生药物相关的属性矩阵`P`
    \n药物属性：
        `disease`
        `pubchem`
        `domain`
        `go`
    '''
    @staticmethod
    def get(property_name: str, extra_information=False) -> Union[np.ndarray, Tuple]:
        file_map = {
            "disease": "drug_disease_mat.csv",
            "pubchem": "drug_pubchem_mat.csv",
            "domain": "drug_target_domain_mat.csv",
            "go": "drug_target_go_mat.csv"
        }
        file_map.update(
            {a+str(i): os.path.join("chemchecker", a+str(i)+".csv")
             for a in ["A", "B", "C", "D", "E"] for i in range(1, 6)}
        )
        assert property_name in file_map.keys(), "Must be in %s" % str(file_map.keys())
        data = load_csv(file_map, property_name)
        if file_map[property_name].startswith("chemchecker"):
            dtype = float
            properties = ["x_"+str(i) for i in range(data.shape[1])]
            drugs = ["r_"+str(i) for i in range(data.shape[0])]
            res = data
        else:
            properties = data[0, 1:]
            drugs = data[1:, 0]
            res = data[1:, 1:]
            dtype = int
        if extra_information:
            return res.astype(dtype), (drugs, properties)
        return res.astype(dtype)


class DiseaseProperty(PropertyFactory):
    '''疾病属性工厂，生产疾病属性矩阵
    \n疾病属性：
        `drug`
    '''
    @staticmethod
    def get(property_name: str, extra_information=False) -> Union[np.ndarray, Tuple]:
        file_map = {
            "drug": "drug_disease_mat.csv"
        }
        assert property_name in file_map.keys(), "Must be in %s" % str(file_map.keys())
        data = load_csv(file_map, property_name)
        if property_name == "drug":
            data = data.transpose()
        properties = data[0, 1:]
        drugs = data[1:, 0]
        if extra_information:
            return data[1:, 1:].astype(int), (drugs, properties)
        return data[1:, 1:].astype(int)
