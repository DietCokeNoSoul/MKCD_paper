import tempfile
import unittest
import os

import numpy as np

from ..data_process.properties import DiseaseProperty, DrugProperty
from ..data_process.similaritys import SimilarityFactory
from ..utils.crossValidation import CrossValidation, DrugDiseaseCrossValidation
from ..metrics.metrics import Metrics
from ..metrics.compare import OtherMethods


class PropertyFactoryTest(unittest.TestCase):

    def test_drug(self):
        for each in ["disease", "pubchem", "domain", "go"]:
            data, (ids, cols) = DrugProperty.get(each, extra_information=True)
            self.assertEqual(data.shape[0], 763)
            self.assertEqual(data.shape[1], len(cols))

    def test_disease(self):
        for each in ["drug"]:
            data, (ids, cols) = DiseaseProperty.get(
                each, extra_information=True)
            self.assertEqual(data.shape[0], 681)
            self.assertEqual(data.shape[1], len(cols))


class SimilarityTest(unittest.TestCase):

    def test_drug(self):
        for each in ["disease", "pubchem", "domain", "go"]:
            s = SimilarityFactory.similarity("drug", each, "Tanimoto")
            self.assertEqual(s.shape[0], s.shape[1])
            self.assertEqual(s.shape[0], 763)
            self.assertTrue(np.all(s.diagonal()-1. < 1e-6))

    def test_disease(self):
        for each in ["drug", "DAG"]:
            s = SimilarityFactory.similarity("disease", each)
            self.assertEqual(s.shape[0], s.shape[1])
            self.assertEqual(s.shape[0], 681)
            self.assertTrue(np.all(s.diagonal()-1. < 1e-6))


class MetricesTest(unittest.TestCase):

    def setUp(self) -> None:
        scores = np.random.rand(1000)
        self.labels = np.random.randint(0, 2, 1000)
        mask = self.labels == 1
        scores[mask] = np.where(
            1-scores[mask] > scores[mask], 1-scores[mask], scores[mask])
        self.scores = scores

    def test_confusion(self):
        cms = Metrics.confusion_matrices(self.scores, self.labels)
        self.assertTrue(np.all(cms >= 0))
        self.assertTrue(np.all(np.sum(cms, axis=(1, 2)) == 1000))

    def test_PR_ROC(self):
        cms = Metrics.confusion_matrices(self.scores, self.labels)
        tpr = Metrics.TPR(cms)
        self.assertEqual(tpr.size, 1000)
        fpr = Metrics.FPR(cms)
        self.assertEqual(fpr.size, 1000)
        p = Metrics.precision(cms)
        self.assertEqual(p.size, 1000-1)
        r = Metrics.recall(cms)
        self.assertEqual(r.size, 1000-1)
        auc = Metrics.auc(r, p)
        self.assertFalse(np.isnan(auc))
        auc = Metrics.auc(fpr, tpr)
        self.assertFalse(np.isnan(auc))


class CrossValidationTest(unittest.TestCase):

    def test_abstract(self):
        cv5 = CrossValidation(5, 888)
        folds = cv5.partition()
        self.assertEqual(len(folds), 5)
        s = sum(map(lambda x: len(x), folds))
        self.assertEqual(s, 888)

    def test_cross_drug_disease(self):
        import tempfile
        RD = DrugProperty.get("disease").astype(int)
        with tempfile.TemporaryDirectory() as dir:
            cv5 = DrugDiseaseCrossValidation(5, RD, dir, 3)
            s = np.prod(RD.shape)
            for i in range(5):
                rd, trains, tests = cv5[i]
                self.assertEqual(s, trains.shape[0]+tests.shape[0])
                self.assertEqual(rd.shape, RD.shape)
                rs, ds = trains[:, 0], trains[:, 1]
                self.assertEqual(rd[rs, ds].sum(), RD[rs, ds].sum())
                scores = np.random.rand(RD.shape[0], RD.shape[1])
                mask = cv5.RD == 1
                scores[mask] = np.where(
                    1-scores[mask] > scores[mask], 1-scores[mask], scores[mask])
                cv5.record_predictions(i, scores)
            fpr, tpr, r, p = cv5.metrics(combine_by_drug=True)


class CompareTest(unittest.TestCase):

    def test_load(self):
        methods = ["CBPred", "LRSSL", "SCMFDD", "HGBI", "MBiRW"]
        metrics = ["TPR", "FPR", "P", "R", "AUC", "AUPR"]
        for m in methods:
            for mm in metrics:
                some = OtherMethods.metrics(m, mm)
                self.assertTrue(some.size > 0)

    def test_plot(self):
        p = np.random.rand(500)
        r = np.random.rand(500)
        tpr = np.random.rand(500)
        fpr = np.random.rand(500)
        r = np.sort(r)
        fpr = np.sort(fpr)
        with tempfile.TemporaryDirectory() as dir:
            OtherMethods.compare("random", r, p, fpr, tpr,
                                 save_fig=os.path.join(dir, "roc.jpg"))
            OtherMethods.topK_recall(
                "random", r, save_fig=os.path.join(dir, "topk.jpg"))


if __name__ == "__main__":
    unittest.main()
