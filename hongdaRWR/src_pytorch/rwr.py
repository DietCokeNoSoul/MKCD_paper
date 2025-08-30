from findFine import *
import matplotlib.pylab as plt
import numpy as np
from utils import joint,rand_walk

def test():
    RD,RRs,DDs=train_data()

    Hs=joint(RRs,DDs,RD)
    n=len(Hs)
    for i,H in enumerate(Hs):
        rds=rand_walk(0.7,3,H)
        fig,axs=plt.subplots(1,8,figsize=(32,4))
        for j,rd in enumerate(rds):
            axs[j].matshow(rd)
            axs[4+j].hist(np.log10(rd[rd!=0]),bins=250)