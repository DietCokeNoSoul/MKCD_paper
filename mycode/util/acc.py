import torch

#计算准确率        
#(模型，测试数据/训练数据,特征矩阵，邻接矩阵，训练集/测试集代表，是否存储,五折交叉哪一折)
def tacc(model,tset,fea,string,s,cros,rwr,device):
    correct=0      #预测正确数
    total=0        #样本总数
    st={0:'train_acc',1:'test_acc'}
    predall,yall=torch.tensor([]).to(device),torch.tensor([]).to(device)     #存预测值和标签
    model.eval()      #测试模式，droupout无效
    for x1,x2,y in tset:
        x1,x2,y=x1.long().to(device),x2.long().to(device),y.long().to(device)
        pred=model(x1,x2,fea,rwr,device,[],0,1).data     #得到预测值(32,2)
        if s==1:
            predall=torch.cat([predall,torch.as_tensor(pred)],dim=0)
            yall=torch.cat([yall,torch.as_tensor(y)])
        a=torch.max(pred,1)[1]#   pred按行取最大值的索引
        # print(a, y)
        total+=y.size(0)    #总数相加
        correct+=(a==y).sum()     #预测对的值相加
    if string==1 and s==1:
        torch.save((predall,yall),'./circ_CNNplt_%d'%cros)
    print(st[string]+str((correct/total).item()))