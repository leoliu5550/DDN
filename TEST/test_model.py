import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from model import *

class TestCNNBlock:
    def test_CNN(self):
        x = torch.ones(1,3,200,200) # 3 times = 56x56
        model = CNNBlock(in_channel=3,out_channel=64,kernel_size = 3,stride = 2)
        assert model(x).shape == torch.Size([1, 64, 99, 99])

class TestResNet:
    def test_BTNK1(self):
        x = torch.ones(1,64,112,112)
        model1 = BTNK1(in_channel=64,stride = 1)
        assert model1(x).shape == torch.Size([1, 256, 112, 112])

        x2 = torch.ones(1,256,56,56) 
        model2 = BTNK1(in_channel=256,half = True,stride = 1)
        assert model2(x2).shape == torch.Size([1, 512, 56, 56])

    def test_BTNK2(self):
        x = torch.ones(1,10,200,200) 
        model = BTNK2(in_channel=10,stride = 1)
        assert model(x).shape == torch.Size([1,10,200,200])

    def test_R1Block(self):
        x = torch.ones(1,3,224,224) 
        model = R1Block(in_channel=3)
        assert model(x).shape == torch.Size([1, 64, 112, 112])

    def test_R2Block(self):
        x = torch.ones(1, 64, 112, 112) 
        model = R2Block(in_channel=64)
        assert model(x).shape == torch.Size([1, 256, 56, 56])
        
    def test_R3Block(self):
        x = torch.ones(1, 256, 56, 56) 
        model = R3Block(in_channel=256)
        assert model(x).shape == torch.Size([1, 512, 28, 28])

    def test_R4Block(self):
        x = torch.ones(1, 512, 28, 28) # 3 times = 56x56
        model = R4Block(in_channel=512)
        assert model(x).shape == torch.Size([1, 1024, 14, 14])

    def test_R5Block(self):
        x = torch.ones(1, 1024, 14, 14) # 3 times = 56x56
        model = R5Block(in_channel=1024)
        assert model(x).shape == torch.Size([1, 2048, 7, 7])


class TestMFN:
    def test_MFN(self):
        model = MFN()
        x = torch.rand(1,3,320,320)
        assert model(x).shape == torch.Size([1,3840,20,20])

