import torch
import torch.nn as nn
import torch.nn.functional as F
from RWR.rwr import combine_rwr

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads # //表示整数除法
        
        self.q_linear = nn.Linear(d_model, self.d_k * num_heads)
        self.k_linear = nn.Linear(d_model, self.d_k * num_heads)
        self.v_linear = nn.Linear(d_model, self.d_v * num_heads)
        self.out = nn.Linear(self.d_v * num_heads, d_model)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.q_linear.bias, 0)
        nn.init.constant_(self.k_linear.bias, 0)
        nn.init.constant_(self.v_linear.bias, 0)
        nn.init.constant_(self.out.bias, 0)
        
    def forward(self, H, rwr, device):
        # Perform linear operation and split into num_heads
        Q = self.q_linear(H).view(-1, self.num_heads, self.d_k)
        K = self.k_linear(H).view(-1, self.num_heads, self.d_k)
        V = self.v_linear(H).view(-1, self.num_heads, self.d_v)
        
        # Transpose to get dimensions num_heads * n * d_k
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)
        
        # Calculate attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        rg = rwr.rg.unsqueeze(0).repeat(scores.shape[0], 1, 1).to(device)
        scores = scores * rg #
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to V
        context = torch.matmul(attention, V) # 注意力权重乘以V
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(0, 1).contiguous().view(-1, self.d_v * self.num_heads) # 这里是将多头注意力的结果拼接起来
        out = self.out(context)
        
        return out
class GraphTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList([MultiHeadSelfAttention(d_model, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.eta_1 = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.eta_2 = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.eta_3 = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.weight = [self.eta_1, self.eta_2, self.eta_3]
        self.gate = nn.Linear(1527 * 2, 1527)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for layer in self.layers:
            layer._reset_parameters()
        nn.init.constant_(self.eta_1, 1)
        nn.init.constant_(self.eta_2, 1)
        nn.init.constant_(self.eta_3, 1)
        
    def forward(self, feature, faeture, device):
        rwr = combine_rwr(self.weight,feature)
        features = feature
        for layer in self.layers:
            features = features + layer(features, rwr,device)
        # gate_input = torch.cat([feature, feature], dim=1)
        # gate_output = torch.sigmoid(self.gate(gate_input))
        # feature = (1 - gate_output) * feature + gate_output * faeture
        return feature