import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
# from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math
import torch
from torch_geometric.data import Data

class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, **kwargs):
        super(HGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads # //表示整数除法
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.att           = None
        
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        '''
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
    def forward(self, node_inp, node_type, edge_index, edge_type):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type] 
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    '''
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    '''
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    '''
                        Get the corresponding input node representations by idx.
                        Add tempotal encoding to source representation (j)
                    '''
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]

                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * gelu(Agg(x)) + x
        '''
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
    
    
    
class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm = True):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'hgt':
            self.base_conv = HGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm)

    def forward(self, meta_xs, node_type, edge_index, edge_type):
        if self.conv_name == 'hgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type)

    


# 读取数据 circRNA-disease5*834*834 覆盖过得特征矩阵5*834*138 训练集索引表5*2*1576 测试集索引表5*2*113315
_, cd, features, trainSet_index, testSet_index = torch.load('circ_CNN.pth') 
print(cd.shape, features[0].shape, trainSet_index[0].shape, testSet_index[0].shape)

# # 生成节点特征
# num_nodes = features[0].shape[0]
# in_dim = features[0].shape[1]
# x = features[0].float()

# # 生成边索引和边类型
# edge_index = []
# edge_type = []
# node_type = []

# # 生成节点类型
# node_type = torch.cat([
#     torch.zeros(834, dtype=torch.long),  # circRNA
#     torch.ones(138, dtype=torch.long),   # disease
#     torch.full((555,), 2, dtype=torch.long)  # miRNA
# ])


# # 从features[0]中取出cda，cda为834*138的矩阵，表示circRNA与disease之间的关系
# cda = features[0][:834, 834:972]
# # # 从features[0]中取出相似矩阵cc，cc为834*834的矩阵，表示circRNA与circRNA之间的关系
# # cc = features[0][:834, :834]
# # # 取出相似矩阵dd，dd为138*138的矩阵，表示disease与disease之间的关系
# # dd = features[0][834:972, 834:972]
# # # 取出相似矩阵mm，mm为555*555的矩阵，表示miRNA与miRNA之间的关系
# # mm = features[0][972:, 972:]
# # 生成边索引和边类型
# for i in range(834):  # circRNA
#     for j in range(138):  # disease
#         if cda[i, j] == 1:
#             edge_index.append([i, 834 + j])
#             edge_type.append(0)  # circRNA-disease
# # for i in range(834):  # circRNA
# #     for j in range(834):
# #         if cc[i, j] > 0:
# #             edge_index.append([i, j])
# #             edge_type.append(1)
# # for i in range(138):  # disease
# #     for j in range(138):
# #         if dd[i, j] > 0:
# #             edge_index.append([834 + i, 834 + j])
# #             edge_type.append(2)
# # for i in range(555):  # miRNA
# #     for j in range(555):
# #         if mm[i, j] > 0:
# #             edge_index.append([972 + i, 972 + j])
# #             edge_type.append(3)

# edge_index = torch.tensor(edge_index).t().contiguous() # contiguous()返回具有相同数据但不同形状的张量
# edge_type = torch.tensor(edge_type)


# # Create a PyG data object
# data = Data(x=x, edge_index=edge_index, node_type=node_type, edge_type=edge_type)

# # Initialize the GeneralConv model
# model = GeneralConv(conv_name='hgt', in_hid=in_dim, out_hid=in_dim, num_types=3, num_relations=4, n_heads=1, dropout=0.2)

# # Forward pass
# output = model(data.x, data.node_type, data.edge_index, data.edge_type)
# print(output)