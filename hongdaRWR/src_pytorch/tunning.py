import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os
import torch
from model import mSACN

dir_preprocess = "./data/processed"
drugs = np.load(os.path.join(dir_preprocess, "drugs.npy"))
diseases = np.load(os.path.join(dir_preprocess, "diseases.npy"))
print("Have %d drugs, %d diseases." % (len(drugs), len(diseases)))

D = np.load(os.path.join(dir_preprocess, "dis_sim.npy"))
RD = np.load(os.path.join(dir_preprocess, "dru_dis.npy"))
R_d = np.load(os.path.join(dir_preprocess, "dru_dom.npy"))
R_g = np.load(os.path.join(dir_preprocess, "dru_go.npy"))
R_c = np.load(os.path.join(dir_preprocess, "dru_che.npy"))


RS_dis = np.load(os.path.join(
    dir_preprocess, "drug_similarity_cosine_disease.npy"))
RS_dom = np.load(os.path.join(
    dir_preprocess, "drug_similarity_cosine_target_domain.npy"))
RS_go = np.load(os.path.join(
    dir_preprocess, "drug_similarity_cosine_target_go.npy"))
RS_che = np.load(os.path.join(
    dir_preprocess, "drug_similarity_cosine_pubchem.npy"))

DS = [D]
RS = [RS_dis, RS_dom, RS_go, RS_che]

print("%d disease similaritys and %d drug similaritys." % (len(DS), len(RS)))


class ConvSet(torch.utils.data.Dataset):
    def __init__(self, RD, RS, DS, indexs, label, gpu_id=0):
        super(ConvSet, self).__init__()
        data = torch.from_numpy(np.stack(
            [
                np.vstack(
                    (
                        np.hstack((r, RD)), np.hstack((RD.T, d))
                    )
                ) for r in RS for d in DS
            ]
        )).float()
        label = torch.from_numpy(label).float()
        self.indexs = indexs
        if gpu_id < 0:
            self.label = label
            self.data = data
        else:
            self.label = label.cuda(gpu_id)
            self.data = data.cuda(gpu_id)

    def __getitem__(self, index):
        r, d = self.indexs[index]
        data = torch.stack([
            self.data[:, r, :],
            self.data[:, self.label.shape[0]+d, :]
        ], dim=1)
        return (r, d), data, self.label[r, d]

    def __len__(self):
        return len(self.indexs)


knowns = RD > 0
unknowns = RD == 0
print("There are %d knowns and %d unknowns in %d pairs." %
      (knowns.sum(), unknowns.sum(), np.prod(RD.shape)))
print("Knowns accounted for %.4f %%." % (knowns.sum()/np.prod(RD.shape)*100))

pairs_known = np.argwhere(knowns)
pairs_unknown = np.argwhere(unknowns)

fold = np.random.randint(5)
print("Using fold %d now." % (fold+1))

indices = np.arange(len(pairs_known))
np.random.shuffle(indices)
n_f = len(indices)//5
start = fold*n_f
end = start+n_f
selected = indices[start:end]
remained = np.concatenate((indices[:start], indices[end+1:]))
mask = np.zeros(RD.shape, bool)
mask[tuple(np.transpose(pairs_known[selected]))] = 1
useful = mask.sum(axis=1) >= 3
print("Remained %d drugs to be tested." % useful.sum())
mask = np.zeros(RD.shape, bool)
mask[useful] = 1
mask[tuple(np.transpose(pairs_known[remained]))] = 0
mask[tuple(np.transpose(pairs_unknown[remained]))] = 0
print("Total %d pairs need to be tested." % mask.sum())
rd = RD.copy()
rd[tuple(np.transpose(pairs_known[selected]))] = 0

train = np.concatenate((pairs_known[remained], pairs_unknown[remained]))
test = np.argwhere(mask)
print("Using %d pairs to train." % len(train))
dataset_train = ConvSet(rd, RS, DS, train, RD)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)


class ConvModel(torch.nn.Module):
    def __init__(self, in_features, num_classes):
        super(ConvModel, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 16, kernel_size=3,
                            stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        )
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(23232, 100),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, num_classes),
            # torch.nn.Sigmoid()
        )

    def forward(self, data):
        x = self.features(data).flatten(1)
        # print(x.shape)
        y = self.classify(x).flatten(0)
        return y


model = ConvModel(len(RS)*len(DS), 1).cuda()
# model = mSACN(len(RS)*len(DS), 1444, [16, "M", 32, "M"]).cuda()
loss = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
dataset_test = ConvSet(rd, RS, DS, test, RD)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1024)


def confusion(r, RD, mask, useful):
    length = mask[useful].sum(axis=1).min()
    mask = mask.copy()
    neg = np.logical_and(mask, RD == 0)
    for i, d in enumerate(useful):
        if not d:
            continue
        delta = mask[i].sum()-length
        if delta == 0:
            continue
        ps = np.argwhere(neg[i])
        np.random.shuffle(ps)
        mask[i, tuple(np.transpose(ps[:delta]))] = 0
    predicts = [r[i][mask[i]] for i, drug in enumerate(useful) if drug]
    targets = [RD[i][mask[i]] for i, drug in enumerate(useful) if drug]
    cs = []
    for target, predict in zip(targets, predicts):
        indices = np.argsort(predict)
        tp = target.sum()
        fp = length-tp
        tn = 0
        fn = 0
        c = [[tp, fp, tn, fn]]
        for i in indices:
            if target[i] == 1:
                tp = tp-1
                fn = fn+1
            else:
                tn = tn+1
                fp = fp-1
            c.append([tp, fp, tn, fn])
        cs.append(c)
    return np.array(cs)


def pr_roc(c):
    tp, fp, tn, fn = 0, 1, 2, 3
    p = c[:, tp]/(c[:, tp]+c[:, fp])
    r = c[:, tp]/(c[:, tp]+c[:, fn])
    tpr = r.copy()
    fpr = c[:, fp]/(c[:, fp]+c[:, tn])
    return p, r, tpr, fpr


epochs = 80
for e in range(epochs):
    R = np.zeros_like(RD, float)
    for i, x, y in loader_train:
        y_hat = model(x)
        l = loss(y_hat, y)
        opt.zero_grad()
        l.backward()
        opt.step()
        y_hat = y_hat.detach().cpu().numpy()
        R[tuple(i)] = y_hat
    # if e <29:continue
    with torch.no_grad():
        for i, x, y in loader_test:
            y_hat = model(x).sigmoid().cpu().numpy()
            R[tuple(i)] = y_hat

        cs = confusion(R, RD, mask, useful)
        c = cs.sum(axis=0)
        p, r, tpr, fpr = pr_roc(c[:-1])
        # print(p)
        print(auc(r, p), auc(fpr, tpr))
        print(R[mask].shape)
        plt.figure()
        plt.scatter(x=np.arange(mask.sum()), y=R[mask], c=RD[mask])
        plt.savefig("44.jpg")
