from torch.optim import Optimizer
from torch.nn import Module,Softmax
from torch.utils.data import DataLoader
from torch import device as cuda_device
from torch import stack,no_grad
from torch.cuda import is_available as is_available
from numpy import concatenate


device=cuda_device('cuda:1') if is_available() else cuda_device('cpu')

def validate(λ:float,m1:Module,m2:Module, dataloader1:DataLoader,dataloader2:DataLoader,device=device):
    outputs=[]
    labels=[]
    softmax=Softmax(dim=-1)
    m1.eval()
    m1=m1.to(device)
    m2.eval()
    m2=m2.to(device)
    dataloader1.dataset.to_device(device)
    dataloader2.dataset.to_device(device)
    with no_grad():
        for x,y in zip(dataloader1,dataloader2):
            input1,target1=x
            input2,_=y
            input1=input1.float()
            target1=target1.long()
            input2=input2.float()
            output1=m1(input1)
            output2=m2(input2)
            output=λ*output1+(1.-λ)*output2
            output=softmax(output)
            outputs.append(output.cpu().numpy())
            labels.append(target1.cpu().numpy())
    return concatenate(outputs,axis=0),concatenate(labels,axis=0)
        
