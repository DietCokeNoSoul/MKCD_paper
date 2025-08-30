# import torch
# import torch.nn as nn
# import os
# import sys
# import torch.nn.functional as F
# # 获取当前文件的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取项目根目录
# project_root = os.path.dirname(current_dir)
# # 将项目根目录添加到 sys.path
# sys.path.append(project_root)
# from efficientKAN.kan import KAN
# # from pykan.kan.MultKAN import MultKAN
# # from pyHGT.conv import GeneralConv
# from pyHGT.transformer import GraphTransformer
# device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# #网络模型
# class Net(nn.Module):
#     def __init__(self, in_dim=1527, seed = 10):
#         super(Net,self).__init__()
#         #Convolution卷积,2d,两维卷积,(输入通道数,输出通道数,卷积窗口大小,步长,补充几圈0padding)
#         #MaxPool2d(池化窗口大小,步长) 输入为1*2*1527
#         # self.kanconv1 = FasterKANConv2d(1, 32, kernel_size=2, stride=1, padding=1)
#         # self.kanconv2 = FasterKANConv2d(32, 64, kernel_size=2, stride=1, padding=0)
#         # self.kanconv3 = FasterKANConv2d(64, 128, kernel_size=(1, 2), stride=1, padding=0)
#         # self.hgt = GeneralConv(conv_name='hgt', in_hid=in_dim, out_hid=in_dim, num_types=3, num_relations=4, n_heads=1, dropout=0.2)
#         self.transformer = GraphTransformer(d_model=in_dim, num_heads=2, num_layers=1)
#         self.c1 = nn.Conv2d(1, 32, kernel_size=(2,2), stride=1, padding=1) #输出32*3*1527
#         self.p1 = nn.MaxPool2d(kernel_size=(2, 2)) #输出32*1*763
#         self.c2 = nn.Conv2d(32, 64, kernel_size=(1, 2), stride=1, padding=0) #
#         self.c3 = nn.Conv2d(64, 128, kernel_size=(1, 2), stride=1, padding=0) #
#         self.p2 = nn.MaxPool2d(kernel_size=(1, 7)) #
#         self.l1 = nn.Linear(1920, 512) #全连接层
#         self.l2 = nn.Linear(512, 256)
#         self.l3 = nn.Linear(256, 2) #全连接层,输出2个类别
#         # self.c4 = nn.Conv2d(128, 128, kernel_size=(1, 2), stride=1, padding=0) #
#         # self.kan = KAN([3968, 512, 2]) # KAN模型
#         # self.resC = nn.Conv2d(1, 1, kernel_size=(2, 2), stride=1, padding=0)
#         self.resKan = KAN([1527*2, 1024, 256])
#         self.lk1 = nn.Linear(1527*2, 1024)
#         self.lk2 = nn.Linear(1024, 256)
#         # self.multkan = MultKAN([3968, 512, 2]) # MultKAN模型

#         self.leakyrelu = nn.LeakyReLU() #激活函数
#         self.d = nn.Dropout(0.5) #dropout层
#         self.gate = nn.Linear(in_dim * 2, in_dim)
#         self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))
#         self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float32))
#         self.reset_para(seed)
        
#     #权重参数初始化
#     def reset_para(self, seed):
#         nn.init.xavier_normal_(self.c1.weight) #权重初始化,对c1的权重进行初始化 
#         nn.init.xavier_normal_(self.c2.weight) #权重初始化,对c2的权重进行初始化
#         nn.init.xavier_normal_(self.c3.weight) #权重初始化,对c3的权重进行初始化
#         # nn.init.xavier_normal_(self.c4.weight) #权重初始化,对c3的权重进行初始化
#         nn.init.xavier_normal_(self.l1.weight, gain= nn.init.calculate_gain('relu'))
#         nn.init.xavier_normal_(self.l2.weight, gain= nn.init.calculate_gain('relu')) #对l2的权重进行初始化
#         nn.init.xavier_normal_(self.l3.weight) #对l2的权重进行初始化   
#         nn.init.xavier_normal_(self.gate.weight) #对l2的权重进行初始化
#         torch.nn.init.constant_(self.gate.bias, 1) #对l2的权重进行初始化
        
#         #x1 x2  使用的是circ_dis_asso 中的下标 代表几号  circ 和  dis 
#     def forward(self,x1,x2,features,rwr,device,para=[],epoch=0,state=0): # features:1527*1527 行为circRNA, disease, miRNA,列为circRNA, disease, miRNA
#         x = self.transformer(features, rwr,device)
#         gate_input = torch.cat([features, x], dim=1)
#         gate_output = torch.sigmoid(self.gate(gate_input))
#         x = (1 - gate_output) * x + gate_output * features
#         para.append(gate_output)
#         # x = features + x
#         # rwr.rg = rwr.rg.to(device)
#         # x = features*rwr.rg
#         x2 = x2 + 834
#         x = torch.cat([x[x1][:,None,None,:], x[x2][:,None,None,:]], dim=2)     #两个特征按列拼接,x:32*1*2*1527
        
#         # resKan_x = self.leakyrelu(self.resC(x))
        
#         resKan_x = x
#         resKan_x = resKan_x.reshape(resKan_x.shape[0], -1)
#         resKan_x = self.resKan(resKan_x)
#         # resKan_x = self.leakyrelu(self.lk1(resKan_x))
#         # resKan_x = self.d(resKan_x)
#         # resKan_x = self.leakyrelu(self.lk2(resKan_x))
#         # resKan_x = self.d(resKan_x)
        
#         x = self.leakyrelu(self.c1(x))
#         x = self.p1(x)
#         x = self.leakyrelu(self.c2(x))
#         x = self.p2(x)
#         x = self.leakyrelu(self.c3(x))
#         x = self.p2(x)
        
#         # 可学习参数的和为1
#         alpha_beta = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)
#         self.alpha.data = alpha_beta[0].data
#         self.beta.data = alpha_beta[1].data
        
#         #变成二维,第一维32(batch)不变,第二维即全部展开
#         x = x.reshape(x.shape[0], -1)
#         # x = self.alpha * x + self.beta * resKan_x
        
#         x = self.leakyrelu(self.l1(x))
#         x = self.d(x)
#         x = self.leakyrelu(self.l2(x))
#         x = self.alpha * x + self.beta * resKan_x
#         x = self.d(x)
#         x = self.l3(x)
#         # x = self.kan(x)
#         # x = self.multkan(x)
#         return x #返回的是一个2维的tensor,第一维是batch,第二维是2,代表两个类别的概率


from queue import Full
import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F
from RWR.rwr import combine_rwr
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)
from efficientKAN.kan import KAN
# from pykan.kan.MultKAN import MultKAN
# from pyHGT.conv import GeneralConv
from pyHGT.transformer import GraphTransformer
device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

#网络模型
class Net(nn.Module):
    def __init__(self, in_dim=1527):
        super(Net,self).__init__()
        #Convolution卷积,2d,两维卷积,(输入通道数,输出通道数,卷积窗口大小,步长,补充几圈0padding)
        #MaxPool2d(池化窗口大小,步长) 输入为1*2*1527
        # self.kanconv1 = FasterKANConv2d(1, 32, kernel_size=2, stride=1, padding=1)
        # self.kanconv2 = FasterKANConv2d(32, 64, kernel_size=2, stride=1, padding=0)
        # self.kanconv3 = FasterKANConv2d(64, 128, kernel_size=(1, 2), stride=1, padding=0)
        # self.hgt = GeneralConv(conv_name='hgt', in_hid=in_dim, out_hid=in_dim, num_types=3, num_relations=4, n_heads=1, dropout=0.2)
        self.transformer = GraphTransformer(d_model=in_dim, num_heads=2, num_layers=1)
        self.c1 = nn.Conv2d(1, 32, kernel_size=(2,2), stride=1, padding=1) #输出32*3*1527
        self.p1 = nn.MaxPool2d(kernel_size=(2, 2)) #输出32*1*763
        self.c2 = nn.Conv2d(32, 64, kernel_size=(1, 2), stride=1, padding=0) #
        self.c3 = nn.Conv2d(64, 128, kernel_size=(1, 2), stride=1, padding=0) #
        self.p2 = nn.MaxPool2d(kernel_size=(1, 7)) #
        self.l1 = nn.Linear(1920, 512) #全连接层
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 2) #全连接层,输出2个类别
        # self.c4 = nn.Conv2d(128, 128, kernel_size=(1, 2), stride=1, padding=0) #
        # self.kan = KAN([3968, 512, 2]) # KAN模型
        # self.resC = nn.Conv2d(1, 1, kernel_size=(2, 2), stride=1, padding=0)
        self.resKan = KAN([1527*2, 1024, 256])
        self.eta_1 = nn.Parameter(torch.rand(1, dtype=torch.float32))
        self.eta_2 = nn.Parameter(torch.rand(1, dtype=torch.float32))
        self.eta_3 = nn.Parameter(torch.rand(1, dtype=torch.float32))
        self.weight = [self.eta_1, self.eta_2, self.eta_3]
        # self.multkan = MultKAN([3968, 512, 2]) # MultKAN模型

        self.leakyrelu = nn.LeakyReLU() #激活函数
        self.d = nn.Dropout(0.5) #dropout层
        self.gate = nn.Linear(in_dim * 2, in_dim)
        self.alpha = nn.Parameter(torch.rand(1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.rand(1, dtype=torch.float32))
        # self.reset_para(seed)
        
    #权重参数初始化
    def reset_para(self, seed):
        nn.init.xavier_normal_(self.c1.weight) #权重初始化,对c1的权重进行初始化 
        nn.init.xavier_normal_(self.c2.weight) #权重初始化,对c2的权重进行初始化
        nn.init.xavier_normal_(self.c3.weight) #权重初始化,对c3的权重进行初始化
        # nn.init.xavier_normal_(self.c4.weight) #权重初始化,对c3的权重进行初始化
        nn.init.xavier_normal_(self.l1.weight, gain= nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.l2.weight, gain= nn.init.calculate_gain('relu')) #对l2的权重进行初始化
        nn.init.xavier_normal_(self.l3.weight) #对l2的权重进行初始化   
        nn.init.xavier_normal_(self.gate.weight) #对l2的权重进行初始化
        torch.nn.init.constant_(self.gate.bias, seed) #对l2的权重进行初始化
        
        #x1 x2  使用的是circ_dis_asso 中的下标 代表几号  circ 和  dis 
    def forward(self,x1,x2,features,rwr,device,para=[],epoch=0,state=0): # features:1527*1527 行为circRNA, disease, miRNA,列为circRNA, disease, miRNA
        x = features
        # x = self.transformer(features, rwr,device)

        # self.weights = torch.softmax(torch.stack(self.weight), dim=0)
        # rwr_f = combine_rwr(self.weight, x)
        # x = features*rwr_f.rg
        # x = x*rwr
        x2 = x2 + 834
        # if(state == 0):
        #     para[epoch].append(self.weights)
        x = torch.cat([x[x1][:,None,None,:], x[x2][:,None,None,:]], dim=2)     #两个特征按列拼接,x:32*1*2*1527
        # resKan_x = x
        # resKan_x = resKan_x.reshape(resKan_x.shape[0], -1)
        # resKan_x = self.resKan(resKan_x)
        
        # # 可学习参数的和为1
        # alpha_beta = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)
        # self.alpha.data = alpha_beta[0].data
        # self.beta.data = alpha_beta[1].data
        
        x = self.leakyrelu(self.c1(x))
        x = self.p1(x)
        x = self.leakyrelu(self.c2(x))
        x = self.p2(x)
        x = self.leakyrelu(self.c3(x))
        x = self.p2(x)
        #变成二维,第一维32(batch)不变,第二维即全部展开
        x = x.reshape(x.shape[0], -1)
        x = self.leakyrelu(self.l1(x))
        x = self.d(x)
        x = self.leakyrelu(self.l2(x))
        # x = self.alpha * x + self.beta * resKan_x
        # 保存参数，将alpha和beta的值存入para列表，每一列代表一组参数
        # if(para != []):
        #     para.append([self.alpha.item(), self.beta.item()])
        x = self.d(x)
        x = self.l3(x)
        return x #返回的是一个2维的tensor,第一维是batch,第二维是2,代表两个类别的概率