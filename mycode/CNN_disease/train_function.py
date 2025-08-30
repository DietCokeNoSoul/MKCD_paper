import time
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
def train(model,cost,optimizer,trainSet,testSet,features,epoch,cross, scheduler, rwr,device,para):
    start_time_train = time.time()  # 记录开始时间
    features=features.float().to(device)   #因为网络中W，b的类型常常是float，所以设置输入也为float,放入cuda

    # # 生成边索引和边类型
    # edge_index = []
    # edge_type = [] 
    # node_type = []

    # # 生成节点类型
    # node_type = torch.cat([
    #     torch.zeros(834, dtype=torch.long),  # circRNA
    #     torch.ones(138, dtype=torch.long),   # disease
    #     torch.full((555,), 2, dtype=torch.long)  # miRNA
    # ]).to(device)

    # # 从features[0]中取出cda，cda为834*138的矩阵，表示circRNA与disease之间的关系
    # cda = features[:834, 834:972]
    # # # 从features[0]中取出相似矩阵cc，cc为834*834的矩阵，表示circRNA与circRNA之间的关系
    # # cc = features[:834, :834]
    # # # 取出相似矩阵dd，dd为138*138的矩阵，表示disease与disease之间的关系
    # # dd = features[834:972, 834:972]
    # # # 取出相似矩阵mm，mm为555*555的矩阵，表示miRNA与miRNA之间的关系
    # # mm = features[972:, 972:]
    # # 生成边索引和边类型
    # for i in range(834):  # circRNA
    #     for j in range(138):  # disease
    #         if cda[i, j] == 1:
    #             edge_index.append([i, 834 + j])
    #             edge_type.append(0)  # circRNA-disease
    # # for i in range(834):  # circRNA
    # #     for j in range(834):
    # #         if cc[i, j] == 1:
    # #             edge_index.append([i, j])
    # #             edge_type.append(1)
    # # for i in range(138):  # disease
    # #     for j in range(138):
    # #         if dd[i, j] == 1:
    # #             edge_index.append([834 + i, 834 + j])
    # #             edge_type.append(2)
    # # for i in range(555):  # miRNA
    # #     for j in range(555):
    # #         if mm[i, j] ==1:
    # #             edge_index.append([972 + i, 972 + j])
    # #             edge_type.append(3)


    # edge_index = torch.tensor(edge_index).t().contiguous().to(device) # contiguous()返回具有相同数据但不同形状的张量
    # edge_type = torch.tensor(edge_type).to(device)
    
    isSave=0         #是否存储结果，最后一层再存，存储测试集预测结果和标签
    for i in range(epoch):
        running_loss = 0.0 # 每个epoch的损失
        model.train()   #训练模式，droupout有效
        for x1,x2,label in trainSet:
            x1,x2,label=x1.long().to(device),x2.long().to(device),label.long().to(device)   #数据放入cuda,声明为float类型
            out=model(x1,x2,features,rwr,device,para,i)   #训练
            loss=cost(out,label)        #计算损失
            optimizer.zero_grad()       #梯度清零
            loss.backward()             #反向传播
            optimizer.step()            #更新参数
            running_loss += loss.item()
        print(f"Epoch {i+1}, Loss: {running_loss}")
        #最后一轮进行存储，存储测试集的预测结果和标签
        if i+1==epoch:
            # end_time_train = time.time()  # 记录结束时间
            # print("训练时间: {:.2f}毫秒".format((end_time_train - start_time_train) * 1000))
            isSave=1
            print('epoch:%d'%(i+1))           
            acc.tacc(model,trainSet,features,0,isSave,cross,rwr,device)      #训练集准确率
            # start_time_test = time.time()  # 记录开始时间
            acc.tacc(model,testSet,features,1,isSave,cross,rwr,device)        #测试集准确率
            # end_time_test = time.time()
            # print("测试时间: {:.2f}毫秒".format((end_time_test - start_time_test) * 1000))
                # 在每个 epoch 结束后，使用验证集的损失来更新调度器
                # val_loss = evaluate(model,cost,testSet,features, rwr)
                # scheduler.step(val_loss)  #更新学习率
                # scheduler.step()
                # torch.cuda.empty_cache()

# 定义验证集评估函数
def evaluate(model, cost, testSet, features, rwr):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x1,x2,label in testSet:
            x1,x2,label=x1.long().to(device),x2.long().to(device),label.long().to(device)   #数据放入cuda,声明为float类型
            out=model(x1,x2,features,rwr)   #训练
            loss=cost(out,label)        #计算损失
            val_loss += loss.item()
    return val_loss / len(testSet)