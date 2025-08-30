import torch
from torch import nn
# 超参数
learning_rate = 0.0005

epoch = 40

batch_size = 32


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #Convolution卷积,2d,两维卷积,(输入通道数,输出通道数,卷积窗口大小,步长,补充几圈0padding)
        #MaxPool2d(池化窗口大小,步长) 输入为1*2*1527
        self.c1 = nn.Conv2d(1, 32, kernel_size=(2,2), stride=1, padding=0) 
        self.p1 = nn.MaxPool2d(kernel_size=(1, 2)) #
        self.c2 = nn.Conv2d(32, 64, kernel_size=(1, 2), stride=1, padding=0) #
        self.c3 = nn.Conv2d(64, 128, kernel_size=(1, 2), stride=1, padding=0) #
        self.p2 = nn.MaxPool2d(kernel_size=(1, 7)) #
        self.l1 = nn.Linear(1920, 1024) #全连接层
        self.l2 = nn.Linear(1024, 256)
        self.l3 = nn.Linear(256, 2) #全连接层,输出2个类别
        self.leakyrelu = nn.LeakyReLU() #激活函数
        self.d = nn.Dropout(0.5) #dropout层
        
        self.reset_para()
        
    #权重参数初始化z
    def reset_para(self):
        nn.init.xavier_normal_(self.c1.weight) #权重初始化,对c1的权重进行初始化
        nn.init.xavier_normal_(self.c2.weight) #权重初始化,对c2的权重进行初始化
        nn.init.xavier_normal_(self.l1.weight, gain= nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.l2.weight, gain= nn.init.calculate_gain('relu')) #对l2的权重进行初始化
        nn.init.xavier_normal_(self.l2.weight) #对l2的权重进行初始化   
        
        #x1 x2  使用的是circ_dis_asso 中的下标 代表几号  circ 和  dis 
    def forward(self,x1,x2,features): # features:1527*1527 行为circRNA, disease, miRNA,列为circRNA, disease, miRNA
        # x1 circ_feature    x2+834  dis_feature  （想一想邻接矩阵）
        x2 = x2 + 834
        x = torch.cat([features[x1][:,None,None,:], features[x2][:,None,None,:]], dim=2)     #两个特征按列拼接,x:32*1*2*1527
        x = self.p1(self.leakyrelu(self.c1(x)))   
        x = self.p2(self.leakyrelu(self.c2(x)))  
        x = self.p2(self.leakyrelu(self.c3(x)))
        #变成二维,第一维32(batch)不变,第二维即全部展开
        x = x.reshape(x.shape[0], -1) 
        x = self.leakyrelu(self.l1(x))
        x = self.d(x)
        x = self.leakyrelu(self.l2(x))
        x = self.d(x)
        x = self.l3(x)
        return x #返回的是一个2维的tensor,第一维是batch,第二维是2,代表两个类别的概率