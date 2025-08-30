import unittest

import numpy as np
import matplotlib.pyplot as plt

from ..data_process.similaritys import SimilarityFactory
from ..data_process.properties import DrugProperty
from ..methods.BiRW import *

class SimilarityTest(unittest.TestCase):

    def setUp(self) -> None:
        self.sim = SimilarityFactory.similarity("drug", "pubchem","tanimoto")
        rd = DrugProperty.get("disease")
        self.mask = (rd@rd.T) > 0
        

    def test_count_range(self):
        for i in range(10):
            s=Fisher_Yates_shuffling(self.sim)
            ans=count_range(s,self.mask)
            self.assertTrue(ans.size==10)

    def test_hist(self):
        ans=analysis_of_similarity(self.sim,self.mask,7)
        self.assertTrue(ans[0].size==10)
        self.assertTrue(ans[1].size==10)
        labels=["0~0.1","0.1~0.2","0.2~0.3","0.3~0.4",
        "0.4~0.5","0.5~0.6","0.6~0.7","0.7~0.8","0.8~0.9","0.9~1"]
        fig,ax=plt.subplots()
        ax.bar(range(10),ans[0]*100,0.4,label="origin",color='tab:blue')
        ax.plot(range(10),ans[1]*100,label="randomized",color='tab:orange')
        ax.set_xlabel("Drug pairs similarity bins")
        ax.set_ylabel("Percentage of drug pairs sharing diseases")
        ax.legend()
        ax.set_xticks(range(10),labels,rotation=15)
        fig.tight_layout()
        plt.show()


class MBiRWTest(unittest.TestCase):

    def test_MBiRW(self):
        R = SimilarityFactory.similarity("drug", "pubchem","tanimoto")
        D = SimilarityFactory.similarity("disease","DAG","Cosine")
        A = DrugProperty.get("disease")
        RD=MBiRW(R,D,A)
        self.assertTrue(RD.shape==A.shape)


if __name__ == "__main__":
    unittest.main()
