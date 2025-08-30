import torch
import sys
import os
import torch

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

# 导入自定义包
from util import acc


torch.cuda.empty_cache()
device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

#参数（网络，训练数据1576，测试数据11315，训练集特征矩阵1527*1527，伦次，学习率）
def train(model,cost,optimizer,trainSet,testSet,features,epoch,cross):
    features=features.float().to(device)   #因为网络中W，b的类型常常是float，所以设置输入也为float,放入cuda
    isSave=0         #是否存储结果，最后一层再存，存储测试集预测结果和标签
    for i in range(epoch):
        running_loss = 0.0 # 每个epoch的损失
        model.train()   #训练模式，droupout有效
        for x1,x2,label in trainSet:
            x1,x2,label=x1.long().to(device),x2.long().to(device),label.long().to(device)   #数据放入cuda,声明为float类型
            out=model(x1,x2,features)   #训练
            loss=cost(out,label)        #计算损失
            optimizer.zero_grad()       #梯度清零
            loss.backward()             #反向传播
            optimizer.step()            #更新参数
            running_loss += loss.item()
        print(f"Epoch {i+1}, Loss: {running_loss}")
        #最后一轮进行存储，存储测试集的预测结果和标签
        if i+1==epoch:
            isSave=1
            print('epoch:%d'%(i+1))           
            acc.tacc(model,trainSet,features,0,isSave,cross)      #训练集准确率
            acc.tacc(model,testSet,features,1,isSave,cross)        #测试集准确率
        torch.cuda.empty_cache()
