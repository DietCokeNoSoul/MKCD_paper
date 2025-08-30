import os
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)
import unittest
from methods.deepwalk import *
from data_process.properties import DrugProperty
from data_process.similaritys import SimilarityFactory

class DeepWalkTest(unittest.TestCase):

    def test_model(self):
        R=SimilarityFactory.similarity("drug","pubchem")
        D=SimilarityFactory.similarity("disease","DAG")
        RD=DrugProperty.get("disease")
        line_1=np.hstack((R,RD))
        line_2=np.hstack((RD.T,D))
        G=np.vstack((line_1,line_2))
        m=deep_walk(G,2,32,30,6)
        print("done")
        

if __name__ == '__main__':
    DeepWalkTest().test_model()