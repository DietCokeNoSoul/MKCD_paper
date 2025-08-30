import unittest
import torch

from ..methods.MTRD import *

class AttentionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        attrs=6
        steps=3
        batch_size=32
        cls.X=torch.rand(batch_size,steps,attrs,2,1444)
        cls.Y=torch.rand(batch_size,steps,2,1444)

    
    def test_attrs(self):
        attrs=6
        steps=3
        batch_size=32
        out=AttrAttention(attrs,1444,1444//4)(self.X)
        self.assertTrue(out.shape==(batch_size,steps,attrs,2,1444))
        isok=out.sum()==batch_size*steps*2*1444
        self.assertTrue(isok)

    def test_scales(self):
        steps=3
        batch_size=32
        out=ScaleAttention(steps,1444,120)(self.Y)
        self.assertTrue(out.shape==(batch_size,steps,1,1))
        self.assertTrue(out.sum()-batch_size<=1e-4)

class MTRDTest(unittest.TestCase):
    def test_NTR(self):
        attrs=6
        steps=3
        batch_size=32
        X=torch.rand(batch_size,steps,attrs,2,1444)
        model=NTR(steps,attrs,1444,256)
        out=model(X)
        self.assertTrue(out.shape==(batch_size,256))

    def test_MAR(self):
        attrs=6
        batch_size=32
        X=torch.rand(batch_size,attrs,2,1444)
        model=MAR(attrs,1444,256)
        out=model(X)
        self.assertTrue(out.shape==(batch_size,256))

    def test_MTRD(self):
        attrs=6
        steps=3
        batch_size=32
        X_ntr=torch.rand(batch_size,steps,attrs,2,1444)
        X_mar=torch.rand(batch_size,attrs,2,1444)
        model=MTRD(attrs,steps,1444,128)
        y,(y_ntr,y_mar)=model(X_ntr,X_mar)
        self.assertTrue(y.shape==(batch_size,2))
        self.assertTrue(y_ntr.shape==(batch_size,2))
        self.assertTrue(y_mar.shape==(batch_size,2))

class DataMakerTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls) -> None:
        cls.Rs=[np.random.rand(30,30) for i in range(3)]
        cls.Ds=[np.random.rand(40,40) for i in range(4)]
        cls.RD=np.random.rand(30,40)

    def test_networks(self):
        nets=heterogeneous_networks(self.Rs,self.Ds,self.RD)
        self.assertTrue(nets.shape==(12,70,70))

    def test_random_walks(self):
        nets=np.random.rand(12,70,70)
        rar=random_walk_with_restart(nets,6,0.8)
        self.assertTrue(rar.shape==(6,12,70,70))
        shape=network_to_node_attrs(rar).shape
        self.assertTrue(shape==(70,6,12,70))

    def test_dataloader(self):
        nodes=np.random.rand(70,6,12,70)
        nets=torch.from_numpy(nodes)
        X=nets[[1,2,3,4],:,:,:]
        self.assertTrue(X.shape==(4,6,12,70))