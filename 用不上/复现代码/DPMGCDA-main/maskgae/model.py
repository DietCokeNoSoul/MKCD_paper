import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GraphConv
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader


# 损失函数
def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss


# 边索引变为邻接矩阵
def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to(edge_index.device)


# 编码器
class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, layer="gcn",):

        super().__init__()
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels

            self.convs.append(GCNConv(first_channels, second_channels)) # 489压缩到128

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = to_sparse_tensor(edge_index, x.size(0)) # 边索引变为邻接矩阵
        x = x.float()
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.activation(x)
        return x

    @torch.no_grad()
    def get_embedding(self, x, edge_index, mode="cat"):

        self.eval()
        assert mode in {"cat", "last"}, mode

        x = self.create_input_feat(x)
        edge_index = to_sparse_tensor(edge_index, x.size(0))
        out = []
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.activation(x)
            out.append(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.activation(x)
        out.append(x)

        if mode == "cat":
            embedding = torch.cat(out, dim=1)
        else:
            embedding = out[-1]

        return embedding


# 解码器
class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(self, in_channels, hidden_channels, out_channels=1,num_layers=2, dropout=0.5, activation='relu'):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=False):
        x = z[edge[0]] * z[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


# 同构图卷积
class GConv(nn.Module):
    def __init__(self,in_size,feature_size):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.conv= GraphConv(in_size,feature_size,activation=F.relu)
    def forward(self, g, feat):
        return self.conv(g,feat)

# 同构图视角在编码器前的特征学习
class FeatureExtracter(nn.Module):
    def __init__(self, in_size,feature_size):
        super(FeatureExtracter, self).__init__()
        self.homo_layers = nn.ModuleList()
        for i in range(0, len(feature_size)):
            self.homo_layers.append(GConv(in_size[i],feature_size[i]))

    def forward(self, homo_graph, origin_feature):
        homo_feature_c = self.homo_layers[0](homo_graph[0],origin_feature[0])
        homo_feature_d = self.homo_layers[1](homo_graph[1], origin_feature[1])
        return homo_feature_c,homo_feature_d


class DPMGCDA(nn.Module):
    def __init__(self, encoder, edge_decoder, mask=None, feature_extracter=None):
        super().__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.mask = mask
        self.feature_extracter= feature_extracter
        self.loss_fn = ce_loss
        self.negative_sampler = negative_sampling

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.edge_decoder.reset_parameters()


    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def train_step(self, data, optimizer, homo_graph, node_features, batch_size=2 ** 16, grad_norm=1.0):
        self.train()

        if self.feature_extracter==None:
            c_embedding, d_embedding = node_features[0], node_features[1]
        else:
            c_embedding, d_embedding = self.feature_extracter(homo_graph, node_features)

        data.x = torch.cat((c_embedding, d_embedding), dim=0)
        x, edge_index = data.x, data.edge_index
        remaining_edges, masked_edges = self.mask(edge_index) # edge_index[0]：[2, 4130]
        
        loss_total = 0.0
        # neg_edges = edge_index[1]
        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(aug_edge_index, num_nodes=data.num_nodes, num_neg_samples=masked_edges.view(2, -1).size(1)).view_as(masked_edges)

        # print(remaining_edges.shape, masked_edges.shape, neg_edges.shape)
        
        # aug_edge_index, _ = add_self_loops(edge_index)
        # neg_edges = self.negative_sampler(aug_edge_index, num_nodes=data.num_nodes, num_neg_samples=masked_edges.view(2, -1).size(1)).view_as(masked_edges)

        for perm in DataLoader(range(masked_edges.size(1)), batch_size=batch_size, shuffle=True):

            optimizer.zero_grad()

            # print(x.shape, remaining_edges.shape)
            z = self.encoder(x, remaining_edges)

            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]

            # print(batch_masked_edges.shape, batch_neg_edges.shape)

            # ******************* loss for edge reconstruction *********************
            pos_out = self.edge_decoder(z, batch_masked_edges, sigmoid=False)
            neg_out = self.edge_decoder(z, batch_neg_edges, sigmoid=False)
            # print(pos_out.shape, neg_out.shape)
            loss = self.loss_fn(pos_out, neg_out)
            # **********************************************************************

            loss.backward()

            if grad_norm > 0:
                # gradient clipping
                nn.utils.clip_grad_norm_(self.parameters(), grad_norm)

            optimizer.step()

            loss_total += loss.item()
        return loss_total

    @torch.no_grad()
    def batch_predict(self, z, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            preds += [self.edge_decoder(z, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test_step(self, data, index, homo_graph, node_features,batch_size=2**16):
        self.eval()

        if self.feature_extracter == None:
            c_embedding, d_embedding = node_features[0], node_features[1]
        else:
            c_embedding, d_embedding = self.feature_extracter(homo_graph, node_features)

        data.x = torch.cat((c_embedding, d_embedding), dim=0)
        
        edge_index = data.edge_index
        pos_edge_index = index[0]
        neg_edge_index = index[1]

        z = self(data.x, edge_index)
             
        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))

        y = torch.cat([pos_y, neg_y], dim=0)

        return y,pred


    @torch.no_grad()
    def test_step_ogb(self, data, evaluator, pos_edge_index, neg_edge_index, batch_size=2**16):
        self.eval()
        z = self(data.x, data.edge_index)
        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)

        results = {}
        for K in [20, 50, 100]:
            evaluator.K = K
            hits = evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred, })[f"hits@{K}"]
            results[f"Hits@{K}"] = hits

        return results

