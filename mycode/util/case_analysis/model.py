import math
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from torch import einsum
from einops import rearrange, repeat
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')

    # GCN HGNN 参数
    parser.add_argument('--latdim', default=1546, type=int, help='embedding size')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')

    parser.add_argument('--droprate', default=0.5, type=float, help='rate for dropout')
    parser.add_argument('--hyperNum', default=32, type=int, help='number of hyper edges')
    parser.add_argument('--node_type_dim', default=32, type=int)
    parser.add_argument('--drug_num', default=1373, type=int)
    parser.add_argument('--microbe_num', default=173, type=int)
    # Transformer参数
    parser.add_argument('--patch_size', default=50, type=int, help='size of patch')
    parser.add_argument('--attention_heads', default=6, type=int, help='heads of attention layer')
    parser.add_argument('--head_dim', default=50, type=int, help='per heads dim')
    parser.add_argument('--embed_size', default=1550, type=int, help='size of patch')  # 1550 是补0之后的维度
    parser.add_argument('--X_dim', default=50, type=int, help='every patch has X_dim')
    parser.add_argument('--embed_dropout', default=0.2, type=float, help='embed dropout')
    parser.add_argument('--attention_dropout', default=0.2, type=float, help='attention dropout')
    parser.add_argument('--depth_interact_attention', default=1, type=int, help='attention dropout')
    parser.add_argument('--depth_self_attention', default=1, type=int, help='attention dropout')
    parser.add_argument('--mlp_dim', default=512, type=int, help='MLP dim')

    return parser.parse_args()


class interact_Block(nn.Module):
    def __init__(self):
        super(interact_Block, self).__init__()
        self.num_attention_heads = args.attention_heads  # 8
        self.attention_head_size = args.head_dim  # 50
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 8 x 50

        self.q = nn.Linear(args.X_dim, self.all_head_size)
        self.k = nn.Linear(args.X_dim, self.all_head_size)
        self.v = nn.Linear(args.X_dim, self.all_head_size)

        self.norm = nn.LayerNorm(args.X_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.attention_dropout)
        self.scale = args.X_dim ** (-0.5)
        self.out = nn.Linear(self.all_head_size, args.X_dim)

    def forward(self, x1, x2):
        X_Drug, X_Microbe = x1, x2  # 32 31 50
        q_Drug, q_Microbe = self.q(X_Drug), self.q(X_Microbe)
        k_Drug, k_Microbe = self.k(X_Drug), self.k(X_Microbe)
        v_Drug, v_Microbe = self.v(X_Drug), self.v(X_Microbe)  # 32x31x(8x50)
        # 32 x 31 x (8x50) -> 32x8x31x50
        q_Drug = rearrange(q_Drug, 'b n (h d) -> b h n d', d=args.head_dim)
        q_Microbe = rearrange(q_Microbe, 'b n (h d) -> b h n d', d=args.head_dim)
        k_Drug = rearrange(k_Drug, 'b n (h d) -> b h n d', d=args.head_dim)
        k_Microbe = rearrange(k_Microbe, 'b n (h d) -> b h n d', d=args.head_dim)
        v_Drug = rearrange(v_Drug, 'b n (h d) -> b h n d', d=args.head_dim)
        v_Microbe = rearrange(v_Microbe, 'b n (h d) -> b h n d', d=args.head_dim)
        # print('q_drug.shape:', q_Drug.shape) # torch.Size([32, 8, 31, 50])
        m1 = einsum('b h i d, b h j d -> b h i j', q_Drug, k_Drug) * self.scale
        m2 = einsum('b h i d, b h j d -> b h i j', q_Drug, k_Microbe) * self.scale
        m3 = einsum('b h i d, b h j d -> b h i j', q_Microbe, k_Microbe) * self.scale
        m4 = einsum('b h i d, b h j d -> b h i j', q_Microbe, k_Drug) * self.scale

        m1, m2, m3, m4 = self.softmax(m1), self.softmax(m2), self.softmax(m3), self.softmax(m4)
        m1, m2, m3, m4 = self.dropout(m1), self.dropout(m2), self.dropout(m3), self.dropout(m4)

        out1 = einsum('b h i j, b h j d -> b h i d', m1, v_Drug)
        out2 = einsum('b h i j, b h j d -> b h i d', m2, v_Microbe)
        out3 = einsum('b h i j, b h j d -> b h i d', m3, v_Microbe)
        out4 = einsum('b h i j, b h j d -> b h i d', m4, v_Drug)
        # 32x8x31x50 -> 32x32x(50x8)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out3 = rearrange(out3, 'b h n d -> b n (h d)')
        out4 = rearrange(out4, 'b h n d -> b n (h d)')
        out1 = self.out(out1)
        out2 = self.out(out2)
        out3 = self.out(out3)
        out4 = self.out(out4)  # 32 31 50

        X_Drug, X_Microbe = (out1 + out2) / 2, (out3 + out4) / 2  # 32 31 50

        return X_Drug, X_Microbe


# Interact Transformer
class InterT(nn.Module):

    def __init__(self):
        super(InterT, self).__init__()

        # 每个节点的embed有多少个patch
        self.patchs = int(args.embed_size / args.patch_size)
        self.dropout = nn.Dropout(args.embed_dropout)
        self.transformer_interact = interact_Block()
        self.linear = nn.Linear(1550, 512)
        self.act = nn.LeakyReLU()

    def to_patch_embed(self, x):
        x = rearrange(x, 'b c h (w p) -> b (h w) (p c)', p=self.patchs)
        x = rearrange(x, 'b p n -> b n p')
        fc = nn.Linear(args.patch_size, args.X_dim).cuda()
        x = fc(x)
        return x

    def forward(self, x1, x2, embeds):
        x3 = x2 + 1373
        # 补0  [1546 1546] -> [1546 1550]
        zero = torch.zeros(size=(args.latdim, 4)).cuda()
        embed = torch.cat([embeds, zero], dim=1)

        x_drug = embed[x1][:, None, None, :]  # 32 1 1 1550
        x_microbe = embed[x3][:, None, None, :]  # 32 1 1 1550

        x_drug = self.to_patch_embed(x_drug)
        x_microbe = self.to_patch_embed(x_microbe)  # 32x31x50

        x_drug = self.dropout(x_drug)
        x_microbe = self.dropout(x_microbe)  # 32x31x50
        for i in range(args.depth_interact_attention):
            x_drug, x_microbe = self.transformer_interact(x_drug, x_microbe)  # 32 1550

        x_1 = rearrange(x_drug, 'b n p -> b (n p)')[:, None, None, :]  # 32 31 50  -> 32 1 1 1550
        x_2 = rearrange(x_microbe, 'b n p -> b (n p)')[:, None, None, :]
        x_Transformer = torch.cat([x_1, x_2], dim=2)  # 32 1 2 1550
        x_Transformer = self.act(self.linear(x_Transformer))
        # x_original = torch.cat([embeds[x1][:, None, None, :], embeds[x3][:, None, None, :]], dim=2)  # 32 1 2 1546
        # x = torch.cat([x_Transformer, x_original], dim=3)  # 32 1 2 3096
        # x = self.cnn(x)

        return x_Transformer  # 32 1 2 512


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(1, 16, kernel_size=(2, 10), stride=1, padding=0)
        self.s1 = nn.MaxPool2d(kernel_size=(1, 10))
        self.c2 = nn.Conv2d(16, 32, kernel_size=(1, 10), stride=1, padding=0)
        self.s2 = nn.MaxPool2d(kernel_size=(1, 10))
        self.leakyrelu = nn.LeakyReLU()
        self.mlp = nn.Sequential(nn.Linear(29 * 32, 300),
                                 nn.LeakyReLU(),
                                 nn.Linear(300, 2)
                                 )

    def forward(self, x):
        # x 32 1 2 2122
        x = self.s1(self.leakyrelu(self.c1(x)))  # 32 1 1 211
        x = self.s2(self.leakyrelu(self.c2(x)))  # 32 1 1 20
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)

        return x


class FFN_homo_drug(nn.Module):
    def __init__(self):
        super(FFN_homo_drug, self).__init__()
        self.L1 = nn.Linear(1546, 512, bias=True)
        self.L2 = nn.Linear(512, 128, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        x = self.act(x)
        return x


class FFN_homo_mic(nn.Module):
    def __init__(self):
        super(FFN_homo_mic, self).__init__()
        self.L1 = nn.Linear(1546, 512, bias=True)
        self.L2 = nn.Linear(512, 128, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        x = self.act(x)
        return x


class FFN_hete(nn.Module):
    def __init__(self):
        super(FFN_hete, self).__init__()
        self.L1 = nn.Linear(1546, 512, bias=True)
        self.L2 = nn.Linear(512, 128, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        x = self.act(x)
        return x


# 返回原始特征 经过一次、两次GCN的特征
class GCN_homo_drug(nn.Module):
    def __init__(self):
        super(GCN_homo_drug, self).__init__()

        self.w1 = nn.Parameter(torch.randn(1546, 1546))
        self.w2 = nn.Parameter(torch.randn(1546, 1546))
        self.ffn = FFN_homo_drug()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(args.droprate)

    def forward(self, adj, embeds):
        # AXW    1373x1373  1373x1546  1546x1546
        embed1 = self.leakyrelu(adj @ embeds @ self.w1)
        embed2 = self.leakyrelu(adj @ embed1 @ self.w2)
        embed1 = self.leakyrelu(self.ffn(embed1))  # 1546->128
        embed2 = self.leakyrelu(self.ffn(embed2))
        return embed1, embed2


class GCN_homo_mic(nn.Module):
    def __init__(self):
        super(GCN_homo_mic, self).__init__()

        self.w1 = nn.Parameter(torch.randn(1546, 1546))
        self.w2 = nn.Parameter(torch.randn(1546, 1546))
        self.ffn = FFN_homo_mic()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(args.droprate)

    def forward(self, adj, embeds):
        # 173x173 173x1546 1546x512
        embed1 = self.leakyrelu(adj @ embeds @ self.w1)
        embed2 = self.leakyrelu(adj @ embed1 @ self.w2)
        embed1 = self.leakyrelu(self.ffn(embed1))  # 1546->128
        embed2 = self.leakyrelu(self.ffn(embed2))
        return embed1, embed2


class GCN_hete(nn.Module):
    def __init__(self):
        super(GCN_hete, self).__init__()

        self.w1 = nn.Parameter(torch.randn(1546, 1546))
        self.w2 = nn.Parameter(torch.randn(1546, 1546))
        self.ffn = FFN_hete()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(args.droprate)

    def forward(self, adj, embeds):
        embed1 = self.leakyrelu(adj @ embeds @ self.w1)
        embed2 = self.leakyrelu(adj @ embed1 @ self.w2)
        embed1 = self.leakyrelu(self.ffn(embed1))  # 1546->128
        embed2 = self.leakyrelu(self.ffn(embed2))
        return embed1, embed2


class Neighbor_info_integration(nn.Module):
    def __init__(self):
        super(Neighbor_info_integration, self).__init__()
        self.act = nn.LeakyReLU()

    def forward(self, hete_1hop, hete_2hop, drug_homo_1hop, drug_homo_2hop, mic_homo_1hop, mic_homo_2hop, x1, x2):
        # hete_1hop 1546x128  drug_homo_1hop 1373x128   mic_homo_1hop 173x128
        embed_hete = torch.cat([hete_1hop, hete_2hop], dim=1)
        embed_hete_pair = torch.cat([embed_hete[x1][:, None, None, :], embed_hete[x2 + 1373][:, None, None, :]],
                                    dim=2)  # 32 1 2 128

        embed_homo_drug = torch.cat([drug_homo_1hop, drug_homo_2hop], dim=1)  # 1373x256
        embed_homo_drug = embed_homo_drug[x1][:, None, None, :]  # 32 1 1 256
        embed_homo_mic = torch.cat([mic_homo_1hop, mic_homo_2hop], dim=1)  # 173x256
        embed_homo_mic = embed_homo_mic[x2][:, None, None, :]  # 32 1 1 256
        x = torch.cat([embed_homo_drug, embed_homo_mic], dim=2)
        x = torch.cat([x, embed_hete_pair], dim=3)  # 32 1 2 512

        return x


# 返回原始特征和经过超图的特征
class HGNN(nn.Module):
    def __init__(self):
        super(HGNN, self).__init__()
        self.act = nn.LeakyReLU()
        self.Hyper = nn.Parameter(init(torch.empty(args.latdim, args.hyperNum)))  # 超边矩阵

        self.Dnode_embed = nn.Parameter(torch.randn(1, args.node_type_dim))  # 1 32 节点类型特征
        self.Mnode_embed = nn.Parameter(torch.randn(1, args.node_type_dim))  # 1 32
        self.Wd = nn.Parameter(torch.randn(args.drug_num, args.drug_num))
        self.Wm = nn.Parameter(torch.randn(args.microbe_num, args.microbe_num))

        self.embed_weight_HGNN = nn.Parameter(init(torch.empty(1546, 1)))

        self.linear = nn.Linear(1578, 512)

    def forward(self, x1, x2, embeds):
        Hyper = embeds @ self.Hyper  # 1546 32
        Dnode_embed = repeat(self.Dnode_embed, '() e -> n e', n=args.drug_num)  # 1373  32
        Mnode_embed = repeat(self.Mnode_embed, '() e -> n e', n=args.microbe_num)  # 173 32

        Dnode_embed = self.Wd @ Dnode_embed  # 1373 32
        Mnode_embed = self.Wm @ Mnode_embed  # 173 32

        node_type = torch.cat([Dnode_embed, Mnode_embed], dim=0)
        embeds = torch.cat([embeds, node_type], dim=1)  # 附带节点特征的embeds

        hyper_embed = self.act(Hyper @ Hyper.T @ embeds)  # 32 1578

        # 加权重
        hyper_embed = self.embed_weight_HGNN * hyper_embed

        x_drug = hyper_embed[x1][:, None, None, :]  # 32 1 1 1578
        x_microbe = hyper_embed[x2 + 1373][:, None, None, :]  # 32 1 1 1578

        x_microbe = self.act(self.linear(x_microbe))
        x_drug = self.act(self.linear(x_drug))
        x_hyperembed = torch.cat([x_drug, x_microbe], dim=2)

        return x_hyperembed  # 32 1 2 512


class final_model(nn.Module):
    def __init__(self):
        super(final_model, self).__init__()

        self.GCN_homo_drug = GCN_homo_drug()
        self.GCN_homo_mic = GCN_homo_mic()
        self.GCN_hete = GCN_hete()
        self.FFN_homo_drug = FFN_homo_drug()
        self.FFN_homo_mic = FFN_homo_mic()
        self.FFN_hete = FFN_hete()
        self.transformer = InterT()
        self.cnn = CNN()
        self.HGNN = HGNN()
        self.neighbor_info_integration = Neighbor_info_integration()

    def forward(self, x1, x2, embeds, adj_DM, adj_D, adj_M):
        x3 = x2 + 1373
        x_transformer = self.transformer(x1, x2, embeds)  # 32 1 2 1550
        x_HGNN = self.HGNN(x1, x2, embeds)  # 32 1 2 1578

        hete_embed = embeds
        drug_homo_embed = embeds[:1373, :]
        mic_homo_embed = embeds[1373:1546, :]

        hete_1hop, hete_2hop = self.GCN_hete(adj_DM, hete_embed)  # 1546x
        drug_homo_1hop, drug_homo_2hop = self.GCN_homo_drug(adj_D, drug_homo_embed)
        mic_homo_1hop, mic_homo_2hop = self.GCN_homo_mic(adj_M, mic_homo_embed)

        x_neighbor = self.neighbor_info_integration(hete_1hop, hete_2hop, drug_homo_1hop, drug_homo_2hop, mic_homo_1hop,
                                                    mic_homo_2hop, x1, x2)  # 32 1 2 512
        xd_original = embeds[x1][:, None, None, :]
        xm_original = embeds[x3][:, None, None, :]  # 32 1 1 1546
        x_original = torch.cat([xd_original, xm_original], dim=2)  # 32 1 2 1546

        x = torch.cat([x_HGNN, x_transformer, x_neighbor, x_original], dim=3)

        x = self.cnn(x)

        return x


def train(model, train_set, embed, epoch, learn_rate, adj_tri, D_adj, M_adj):
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)
    cost = nn.CrossEntropyLoss()

    embeds = embed.float().cuda()
    adj_tri = adj_tri.float().cuda()
    D_adj = D_adj.float().cuda()
    M_adj = M_adj.float().cuda()

    for i in range(epoch):
        model.train()
        LOSS = 0
        for x1, x2, y in train_set:
            x1, x2, y = x1.long().to(device), x2.long().to(device), y.long().to(device)
            out = model(x1, x2, embeds, adj_tri, D_adj, M_adj)
            loss = cost(out, y)
            LOSS += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: %d / %d Loss: %0.7f" % (i + 1, epoch, LOSS))
        # 如果到最后一轮了，保存测试结果
        if i + 1 == epoch:
            torch.save(model.state_dict(), './pt/best_network.pth')


def test(model, test_set, embeds, adj, D_adj, M_adj):
    predall, yall = torch.tensor([]), torch.tensor([])
    embed = embeds.float().cuda()
    adj = adj.float().cuda()
    D_adj = D_adj.float().cuda()
    M_adj = M_adj.float().cuda()
    model.eval()  # 使Dropout失效
    model.load_state_dict(torch.load('pt/best_network.pth'))
    CAR = []
    print("testing")
    for x1, x2, y in test_set:
        x1, x2, y = x1.long().to(device), x2.long().to(device), y.long().to(device)
        with torch.no_grad():
            pred = model(x1, x2, embed, adj, D_adj, M_adj)
        predall = torch.cat([predall, torch.as_tensor(pred, device='cpu')], dim=0)
    torch.save((predall, yall), './result/result_pre_all')
    print("testing finished")


class MyDataset(Dataset):
    def __init__(self, tri, ld):
        self.tri = tri
        self.ld = ld

    def __getitem__(self, idx):
        x, y = self.tri[idx, :]
        label = self.ld[x][y]
        return x, y, label

    def __len__(self):
        return self.tri.shape[0]


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
args = parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    learn_rate = 0.0005
    epoch = 50
    batch = 32
    # 超边数  边丢弃
    embed, train_index, test_index, double_DM, DM, D, M = torch.load('./embed_index_adj_nomusked.pth')
    net = final_model().to(device)
    train_set = DataLoader(MyDataset(train_index, DM), batch, shuffle=True)
    test_set = DataLoader(MyDataset(test_index, DM), batch, shuffle=False)
    train(net, train_set, embed, epoch, learn_rate, double_DM, D, M)
    test(net, test_set, embed, double_DM, D, M)
