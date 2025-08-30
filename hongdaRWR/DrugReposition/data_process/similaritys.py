import numpy as np
from .properties import *


class PairwiseOperator(object):
    '''接收格式为`(N, d)`的矩阵，返回`(N, N)`的相似性矩阵。
    \n其中`N`为点的个数，`d`为点的特征数。
    '''

    def __call__(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pairwise operator is not implemented.")


class CosineSimilarity(PairwiseOperator):
    '''cosine(x,y)
    \n=(x·y)/(|x||y|)
    \n=(x·y)/(sqrt(|x|*|x|)*sqrt(|y|*|y|))
    '''

    def __call__(self, data: np.ndarray) -> np.ndarray:
        dot = data @ data.T
        modulus = np.expand_dims(np.sqrt(np.diagonal(dot)), 0)
        return dot/(modulus.T @ modulus)


class TanimotoCoefficient(PairwiseOperator):

    def __call__(self, data: np.ndarray) -> np.ndarray:
        dot = data @ data.T
        modulus = np.diag(dot.diagonal())
        ones = np.ones_like(dot)
        adds = ones @ modulus + modulus @ ones
        return dot/(adds - dot)

class JaccordSimilarity(TanimotoCoefficient):

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return super().__call__(data)

class PairwiseOperatorFactory(object):

    @staticmethod
    def create(method: str):
        if method.upper() == "COSINE":
            return CosineSimilarity()
        if method.upper() == "TANIMOTO":
            return TanimotoCoefficient()
        if method.upper() == "JACCARD":
            return JaccordSimilarity()
        else:
            raise NotImplementedError(
                "Method `%s` is not implemented." % method)


class SimilarityFactory(object):

    '''相似性处理工厂

    '''
    @staticmethod
    def similarity(subject: str, property: str, method: str = "Cosine") -> np.ndarray:
        if subject == "drug":
            factory = DrugProperty
        elif subject == "disease":
            factory = DiseaseProperty
            if property == "DAG":
                data = np.loadtxt(os.path.join(
                    data_dir, "disease_similarity_mat.csv"), dtype=str, delimiter=',')
                return data[1:, 1:].astype(float)
        data = factory.get(property_name=property)
        operator = PairwiseOperatorFactory.create(method)
        s = operator(data)
        s[np.isnan(s)] = 0
        s[np.diag_indices_from(s)] = 1
        return s
