import torch
import numpy as np
from torch import nn
from torch.nn import functional


class Encoder(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(Encoder, self).__init__()
        self.Wx = nn.Linear(in_features, out_features)
        self.Wy = nn.Linear(in_features, out_features)

    def forward(self, rd):
        x, y = torch.split(rd, 1, dim=-2)
        x = self.Wx(x)
        x = torch.tanh_(x)
        y = self.Wy(y)
        y = torch.tanh_(y)
        return torch.cat((x, y), dim=-1)


class Decoder(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(Decoder, self).__init__()
        self.Tx = nn.Linear(in_features, out_features)
        self.Ty = nn.Linear(in_features, out_features)

    def forward(self, x):
        r = self.Tx(x)
        d = self.Ty(x)
        return torch.cat((r, d), dim=-2)


class AttrAttention(nn.Module):
    def __init__(self, attrs, in_features, out_features) -> None:
        super(AttrAttention, self).__init__()
        self.WP = Decoder(out_features*2, in_features)
        self.Ws = nn.ModuleList([
            Encoder(in_features, out_features)
            for i in range(attrs)
        ])

    def forward(self, x):
        encodes = []
        for i, rd in enumerate(torch.split(x, 1, dim=-3)):
            en = self.Ws[i](rd)
            en = self.WP(en)
            encodes.append(en)
        encodes = torch.cat(encodes, dim=-3)
        encodes = functional.softmax(encodes, dim=-3)
        return encodes


class ScaleAttention(nn.Module):
    def __init__(self, scales, nodes, hidden) -> None:
        super(ScaleAttention, self).__init__()
        self.W = nn.ModuleList([
            nn.Linear(2*nodes, hidden)
            for i in range(scales)
        ])
        self.H = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=-2)
        alpha = []
        for i, rd in enumerate(torch.split(x, 1, dim=-2)):
            e = self.W[i](rd)
            e = torch.tanh_(e)
            a = self.H(e)
            alpha.append(a)
        alpha = torch.stack(alpha, dim=1)
        alpha = torch.softmax(alpha, dim=1)
        return alpha


class NTR(nn.Module):
    '''Neighboring topological representation'''

    def __init__(self, scales, attrs, nodes, embeddings) -> None:
        super().__init__()
        self.attention_a = AttrAttention(attrs, nodes, nodes//4)
        self.attention_s = ScaleAttention(scales, nodes, nodes//4)
        self.biLSTM = nn.LSTM(
            nodes*2, nodes//4, batch_first=True, bidirectional=True)
        self.trans = nn.Linear(nodes//2*scales, embeddings)

    def forward(self, x):
        x = x*self.attention_a(x)
        x = torch.sum(x, dim=2)
        x = x*self.attention_s(x)
        x = torch.flatten(x, -2)
        x, _ = self.biLSTM(x)
        x = torch.flatten(x, -2)
        x = self.trans(x)
        x = torch.sigmoid(x)
        return x


class MAR(nn.Module):
    '''Multiple attribute representation'''

    def __init__(self, attrs, nodes, embeddings) -> None:
        super().__init__()
        self.cov1_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=attrs,
                out_channels=16,
                kernel_size=(2, 2),
                # dilation=(1, 1),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=2, stride=1, padding=1,),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 2),
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*nodes, embeddings),
        )

    def forward(self, x):
        x1 = self.cov1_1(x)
        x2 = self.conv1(x1)
        x = x1 + x2
        x = self.cov2(x)
        x = self.fc(x)
        return x


class MTRD(nn.Module):
    '''Main model'''

    def __init__(self, attrs, scales, nodes, embeddings) -> None:
        super().__init__()
        self.ntr = NTR(scales, attrs, nodes, embeddings)
        self.mar = MAR(attrs, nodes, embeddings)
        self.fc_ntr = nn.Linear(embeddings, 2)
        self.fc_mar = nn.Linear(embeddings, 2)
        self.fc_cmb = nn.Linear(2*embeddings, 2)

    def forward(self, x_ntr, x_mar):
        ntr = self.ntr(x_ntr)
        mar = self.mar(x_mar)
        cmb = torch.cat((ntr, mar), dim=-1)
        y_ntr = self.fc_ntr(ntr)
        y_mar = self.fc_mar(mar)
        y_cmb = self.fc_cmb(cmb)
        return y_cmb, (y_ntr, y_mar)


def heterogeneous_networks(Rs, Ds, RD):
    networks = []
    for R in Rs:
        for D in Ds:
            up = np.concatenate((R, RD), axis=1)
            down = np.concatenate((RD.T, D), axis=1)
            networks.append(np.concatenate((up, down), axis=0))
    return np.stack(networks, axis=0)


def random_walk_with_restart(networks, steps, alpha):
    paths = [networks]
    A = networks
    S = networks
    for s in range(steps-1):
        S = (1-alpha)*A@S + alpha*A
        paths.append(S)
    return np.stack(paths, axis=0)


def network_to_node_attrs(networks):
    return np.transpose(networks, [3, 0, 1, 2])


