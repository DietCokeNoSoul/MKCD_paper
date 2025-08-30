from torch.optim import Adam,Optimizer
from torch.nn import CrossEntropyLoss,BCELoss,Module
from torch.utils.data import DataLoader
from torch import device as cuda_device
from torch.cuda import is_available as is_available


device=cuda_device('cuda:1') if is_available() else cuda_device('cpu')

def train(m:Module,lr:float,criterion,dataloader:DataLoader,epochs:int,device=device):
    criterion=criterion.to(device)
    m=m.to(device)
    dataloader.dataset.to_device(device)
    opt=Adam(m.parameters(),lr=lr)
    for epoch in range(epochs):
        for input,target in dataloader:
            input=input.float()
            target=target.long()
            output=m(input)
            loss=criterion(output,target)
            opt.zero_grad()
            loss.backward()
            opt.step()