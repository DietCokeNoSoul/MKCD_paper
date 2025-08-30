import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import KFold
import gc

# 示例数据
a = np.random.randint(0, 2, (271, 218))  # 关联矩阵
circ_s = np.random.rand(271, 271)  # circRNA 相似矩阵
drug_s = np.random.rand(218, 218)  # drug 相似矩阵

# 获取正样本和负样本的索引
pos_indices = np.array(np.where(a == 1)).T
neg_indices = np.array(np.where(a == 0)).T

# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储每一折的训练集和测试集
folds = []

for train_index, test_index in kf.split(pos_indices):
    train_pos = pos_indices[train_index]
    test_pos = pos_indices[test_index]
    
    # 随机选择相同数量的负样本
    np.random.shuffle(neg_indices)
    train_neg = neg_indices[:len(train_pos)]
    test_neg = neg_indices[len(train_pos):len(train_pos) + len(test_pos)]
    
    folds.append((train_pos, train_neg, test_pos, test_neg))

def create_data_object(features, a, train_pos, train_neg, test_pos, test_neg):
    # 构建训练集和测试集的掩码
    train_mask = np.zeros(a.shape, dtype=bool)
    test_mask = np.zeros(a.shape, dtype=bool)
    
    train_mask[tuple(train_pos.T)] = True
    train_mask[tuple(train_neg.T)] = True
    test_mask[tuple(test_pos.T)] = True
    test_mask[tuple(test_neg.T)] = True
    
    # 转换为 PyTorch 张量
    x = torch.tensor(features, dtype=torch.float)
    print(a.sum())
    edge_index = torch.tensor(np.array(a.nonzero()), dtype=torch.long)
    y = torch.tensor(a, dtype=torch.long)
    
    # 构建图数据对象
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = torch.tensor(train_mask.flatten(), dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask.flatten(), dtype=torch.bool)
    
    return data

def mask_test_samples(a, test_pos):
    a_masked = a.copy()
    a_masked[tuple(test_pos.T)] = 0
    return a_masked

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(489, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

# 示例：创建第一折的数据对象并进行训练和测试
train_pos, train_neg, test_pos, test_neg = folds[0]

# 遮盖测试集中的正样本
a_masked = mask_test_samples(a, test_pos)

# 构建初始特征矩阵
features_masked = np.zeros((489, 489))
features_masked[:271, :271] = circ_s
features_masked[271:, 271:] = drug_s
features_masked[:271, 271:] = a_masked
features_masked[271:, :271] = a_masked.T

data = create_data_object(features_masked, a_masked, train_pos, train_neg, test_pos, test_neg)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    loss = train(model, data, optimizer)
    print(f'Epoch {epoch}, Loss: {loss}')

accuracy = test(model, data)
print(f'Accuracy: {accuracy}')