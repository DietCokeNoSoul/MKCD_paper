
from os.path import join, exists
from os import mkdir
from numpy import (array, eye, genfromtxt, hstack, matmul, ndarray, savetxt,
                   sqrt, vstack, zeros, save, load, argwhere,matrix)


def get_matrix_from_csv(file_path: str) -> ndarray:
    x = genfromtxt(file_path, delimiter=',',dtype=str)
    print(x[1:,0],x[0,1:])
    return x[1:, 1:].astype(float),x[1:,0],x[0,1:]


def get_matrix_from_npy(file_path: str) -> ndarray:
    return load(file_path)


prefix = '../data/origin'
data_prefix = '../data/processed/'

if not exists(prefix):
    mkdir(prefix)
if not exists(data_prefix):
    mkdir(data_prefix)

files = {
    'dis_sim': 'disease_similarity_mat.csv',
    'dru_dis': 'drug_disease_mat.csv',
    'dru_che': 'drug_pubchem_mat.csv',
    'dru_dom': 'drug_target_domain_mat.csv',
    'dru_go': 'drug_target_go_mat.csv'
}

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


def get_data_from_file(name: str):
    data = None
    data_prefix = '../data/processed/'
    path = join(data_prefix, name+'.npy')
    try:
        data = get_matrix_from_npy(path)
    except:
        print("\tFile %s.npy not found, Creating ... It will last for a while ." % name)
        csv_path = "hongdaRWR\data\origin\\"+ files[name]
        data,_,_= get_matrix_from_csv(csv_path)
        save(path, data)
    return data


def similarity_disease() -> ndarray:
    return get_data_from_file('dis_sim')


def drug_target_Aigo() -> ndarray:
    return get_data_from_file('dru_go')


def drug_target_domain() -> ndarray:
    return get_data_from_file('dru_dom')


def drug_pubchem() -> ndarray:
    return get_data_from_file('dru_che')


def associations_drug_disease() -> ndarray:
    return get_data_from_file('dru_dis')


def Jaccard_similarity(features: matrix) -> matrix:
    '''Jaccard 相似性'''
    s, f = features.shape
    similarity = eye(s)
    for i in range(s):
        for j in range(i):
            x = sum((features[i] == 1) & (features[j] == 1))
            y = sum((features[i] == 0) & (features[j] == 0))
            r = x/(f-y)
            similarity[i, j] = r
            similarity[j, i] = r
    return similarity


def SMC_similarity(features: matrix) -> matrix:
    '''SMC 相似性'''
    s, f = features.shape
    similarity = eye(s)
    for i in range(s):
        for j in range(i):
            x = sum(features[i] == features[j])/f
            similarity[i, j] = x
            similarity[j, i] = x
    return similarity


def Cosine_similarity(features: matrix) -> matrix:
    '''余弦相似性'''
    s, _ = features.shape
    tmp = matmul(features, features.T)
    similarity = eye(s)
    for i in range(s):
        for j in range(i):
            s=sqrt(tmp[i, i]*tmp[j, j])
            x = tmp[i, j]/s if s >0 else 0
            similarity[i, j] = x
            similarity[j, i] = x
    return similarity


def get_diseases():
    try:
        diseases=load(join(data_prefix,"diseases.npy"))
    except:
        path=join(prefix,"drug_disease_mat.csv")
        _,drugs,diseases=get_matrix_from_csv(path)
        save(join(data_prefix,"drugs.npy"),drugs)
        save(join(data_prefix,"diseases.npy"),diseases)
    return diseases


def get_drugs():
    try:
        drugs=load(join(data_prefix,"drugs.npy"))
    except:
        path=join(prefix,"drug_disease_mat.csv")
        _,drugs,diseases=get_matrix_from_csv(path)
        save(join(data_prefix,"drugs.npy"),drugs)
        save(join(data_prefix,"diseases.npy"),diseases)
    return drugs


def similarity_drug(feature, method) -> matrix:
    '''根据药物特征f，使用相似度度量方法f，计算药物相似性'''
    file_name = "drug_similarity_%s_%s" % (method, feature)
    file = join(data_prefix, file_name)
    try:
        similarity = load(file+".npy")
    except FileNotFoundError:
        print("computing the similarity.")
        if feature == "disease":
            X = associations_drug_disease()
        elif feature == "pubchem":
            X = drug_pubchem()
        elif feature == "target_domain":
            X = drug_target_domain()
        elif feature == "target_go":
            X = drug_target_Aigo()
        if method == "jaccard":
            similarity = Jaccard_similarity(X)
        elif method == "cosine":
            similarity = Cosine_similarity(X)
        elif method == "SMC":
            similarity = SMC_similarity(X)
        save(file+".npy", similarity)
        savetxt(file+".csv", similarity, fmt="%1.18f", delimiter=',')
    finally:
        return similarity

if __name__ == '__main__':
    for each in files.keys():
        x=get_data_from_file(each)
        print(f"{each}:shape{x.shape}")
    for f in features:
        for m in methods:
            x=similarity_drug(f,m)
            print(f"drug similarity in {f}, caculated by {m}")
    path=join(prefix,"drug_disease_mat.csv")
    _,drug,disease=get_matrix_from_csv(path)
    save(join(data_prefix,"drugs.npy"),drug)
    save(join(data_prefix,"diseases.npy"),disease)