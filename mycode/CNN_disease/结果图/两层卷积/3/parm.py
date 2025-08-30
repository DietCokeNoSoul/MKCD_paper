# 超参数
learning_rate = 0.0005

epoch = 40

batch_size = 32

'''
self.c = nn.Sequential( # 输入为32*1*2*1527
    nn.Conv2d(1, 32, kernel_size=(2,2), stride=1, padding=1), 
    nn.MaxPool2d(kernel_size=(1, 4)), 
    nn.Conv2d(32, 64, kernel_size=(1, 4), stride=1, padding=0), 
    nn.MaxPool2d(kernel_size=(1, 7))) #卷积层和池化层
self.l1 = nn.Sequential(nn.Linear(3*54*64, 1024), nn.LeakyReLU(), nn.Dropout(0.5), nn.Linear(1024, 256), nn.LeakyReLU()) #全连接层
self.l2 = nn.Sequential(nn.Linear(256, 2)) #全连接层，输出2个类别
self.leakyrelu = nn.LeakyReLU() #激活函数
self.d = nn.Dropout(0.5) #dropout层

x = self.c(x)
#变成二维,第一维32(batch)不变，第二维即全部展开
x = x.reshape(x.shape[0], -1)
x = self.l1(x)
x = self.d(x)
x = self.l2(x)
'''