learning_rate = 0.0005

epoch = 40

batch_size = 32


import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #Convolution卷积,2d,两维卷积,(输入通道数,输出通道数,卷积窗口大小,步长,补充几圈0padding)
        #MaxPool2d(池化窗口大小,步长)
        self.c1 = nn.Conv2d(1, 32, kernel_size=(2,2), stride=1, padding=1) #输入通道数1,输出通道数32,卷积窗口大小2*2,步长1,补充0圈
        self.p1 = nn.MaxPool2d(kernel_size=(2, 2),stride=1) #池化窗口大小1*4,步长1
        self.c2 = nn.Conv2d(32, 64, kernel_size=(2, 2), stride=1, padding=1) #输入通道数32,输出通道数64,卷积窗口大小1*4,步长1,补充0圈
        self.l1 = nn.Sequential(nn.Linear(195456, 1024), nn.LeakyReLU(), nn.Dropout(0.5), nn.Linear(1024, 256), nn.LeakyReLU()) #全连接层
        self.l2 = nn.Sequential(nn.Linear(256, 2)) #全连接层,输出2个类别
        self.leakyrelu = nn.LeakyReLU() #激活函数
        self.d = nn.Dropout(0.5) #dropout层
        
        self.reset_para()
        
    #权重参数初始化
    def reset_para(self):
        nn.init.xavier_normal_(self.c1.weight) #权重初始化,对c1的权重进行初始化
        nn.init.xavier_normal_(self.c2.weight) #权重初始化,对c2的权重进行初始化
        for mode in self.l1: #对l1中的每一层进行初始化
            if isinstance(mode, nn.Linear): #如果是全连接层
                nn.init.xavier_normal_(mode.weight, gain= nn.init.calculate_gain('relu')) #对l1的全连接层权重进行初始化,缩放因子为relu,增益适用于计算Relu激活函数的权重初始化
        nn.init.xavier_normal_(self.l2[0].weight) #对l2的权重进行初始化   
        
        #x1 x2  使用的是circ_dis_asso 中的下标 代表几号  circ 和  dis 
    def forward(self,x1,x2,features): # features:1527*1527 行为circRNA, disease, miRNA,列为circRNA, disease, miRNA
        # x1 circ_feature    x2+834  dis_feature  （想一想邻接矩阵）
        x2 = x2 + 834
        x = torch.cat([features[x1][:,None,None,:], features[x2][:,None,None,:]], dim=2)     #两个特征按列拼接
        x = self.p1(self.leakyrelu(self.c1(x)))   #[32, 64, 1, 162]
        x = self.p1(self.leakyrelu(self.c2(x)))  #[32, 32, 1, 22]
        #变成二维,第一维32(batch)不变,第二维即全部展开
        x = x.reshape(x.shape[0], -1)
        x = self.l1(x)
        x = self.d(x)
        x = self.l2(x)
        return x #返回的是一个2维的tensor,第一维是batch,第二维是2,代表两个类别的概率

net=Net()
fea,x1,x2=torch.randn(1527,1527),torch.linspace(0,31,32).long(),torch.linspace(0,31,32).long()
out=net(x1,x2,fea)