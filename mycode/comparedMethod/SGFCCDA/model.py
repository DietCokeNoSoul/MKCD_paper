import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import train_features_choose, test_features_choose
from scalegcn import BiScaleGCN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_edge_index(matrix):
    # matrix: (n, m) tensor
    # 返回 shape=(2, 边数) 的 LongTensor
    edge_index = (matrix != 0).nonzero(as_tuple=False).t()
    return edge_index



class MLP(nn.Module):
    def __init__(self, embedding_size, drop_rate):
        super(MLP, self).__init__()
        self.embedding_size = embedding_size
        self.drop_rate = drop_rate

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 2, self.embedding_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 4, self.embedding_size // 6),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 6, 1, bias=False),
            nn.Sigmoid()
        ).to(device)
        self.mlp_prediction.apply(init_weights)

    def forward(self, features_embedding):
        predict_result = self.mlp_prediction(features_embedding)
        return predict_result


class Model(nn.Module):
    def __init__(self, circfeat_in, disfeat_in, feature_out, hidden_size, num_layers,
                 dropout, drop_rate, embedding_size, negative_times):
        super(Model, self).__init__()
        self.circfeat_in = circfeat_in
        self.disfeat_in = disfeat_in
        self.feature_out = feature_out
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.embedding_size = embedding_size
        self.negative_times = negative_times

        self.embedding = BiScaleGCN(self.feature_out, self.feature_out, self.hidden_size,
                                  self.num_layers, self.dropout)

        self.W_rna = nn.Parameter(torch.zeros(size=(self.circfeat_in, self.feature_out)))
        self.W_dis = nn.Parameter(torch.zeros(size=(self.disfeat_in, self.feature_out)))

        nn.init.xavier_uniform_(self.W_rna.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_dis.data, gain=1.414)

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.num_layers+1, 1), padding=0),
            nn.ReLU(),
            nn.Flatten()).to(device)

        self.cnn_layer.apply(init_weights)

        self.mlp_prediction = MLP(self.embedding_size, self.drop_rate)

    def forward(self, adj, circ_feature, dis_feature, rel_matrix, train_model, device=device, trainSet_index=None):
        # 输入特征矩阵adj（673*673），circRNA相似度矩阵circ_feature（585*585），disease相似度矩阵dis_feature（88*88），circRNA-disease关联矩阵rel_matrix（585*88）
        circ_circ_f = circ_feature.mm(self.W_rna)
        dis_dis_f = dis_feature.mm(self.W_dis)
        N = circ_circ_f.size()[0] + dis_dis_f.size()[0]

        c_d_feature = torch.cat((circ_circ_f, dis_dis_f), dim=0)
        edge_index = get_edge_index(adj).to(device)

        x = self.embedding(c_d_feature, edge_index, use_softmax=False)
        x = x.view(N, 1, self.num_layers+1, -1)

        cnn_outputs = self.cnn_layer(x).view(N, -1) # 673*1536(结点数*特征维度)

        if train_model:
            # 选择训练集特征和标签
            # train_features_inputs = 
            train_features_inputs, train_lable = train_features_choose(rel_matrix, cnn_outputs, trainSet_index, 788) # 选择训练集特征和标签
            train_mlp_result = self.mlp_prediction(train_features_inputs)
            return train_mlp_result, train_lable
        else:
            test_features_inputs, test_lable = train_features_choose(rel_matrix, cnn_outputs, trainSet_index, 197)
            test_mlp_result = self.mlp_prediction(test_features_inputs)
            return test_mlp_result, test_lable

