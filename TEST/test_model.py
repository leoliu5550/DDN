import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from model import *
'''
should preprocess the image size to 2^n
'''

class TestCNNBlock:
    def test_CNN(self):
        x = torch.ones(1,3,256,256) # 3 times = 56x56
        model = CNNBlock(
            in_channel=3,
            out_channel=64,
            kernel_size = 3,
            padding = 1,
            stride = 2)
        assert model(x).shape == torch.Size([1, 64, 128, 128])

class TestResNet:
    def test_BTNK1(self):
        x = torch.ones(1, 64, 128, 128)
        model1 = BTNK1(in_channel=64,stride = 1)
        assert model1(x).shape == torch.Size([1, 256, 128, 128]) # --> 64 * 4 as half = False

        x2 = torch.ones(1,256,128,128) 
        model2 = BTNK1(in_channel=256,half = True,stride = 1)
        assert model2(x2).shape == torch.Size([1, 512, 128, 128]) # --> 64 * 2 as half = True

    def test_BTNK2(self):
        x = torch.ones(1,10,200,200) 
        model = BTNK2(in_channel=10,stride = 1)
        assert model(x).shape == torch.Size([1,10,200,200])

    def test_R1Block(self):
        x = torch.ones(1,3,256,256) 
        model = R1Block(in_channel=3)
        assert model(x).shape == torch.Size([1, 64, 128, 128])

    def test_R2Block(self):
        x = torch.ones(1, 64, 128, 128) 
        model = R2Block(in_channel=64)
        assert model(x).shape == torch.Size([1, 256, 64, 64])
        
    def test_R3Block(self):
        x = torch.ones(1, 256, 64, 64) 
        model = R3Block(in_channel=256)
        assert model(x).shape == torch.Size([1, 512, 32, 32])

    def test_R4Block(self):
        x = torch.ones(1, 512, 32, 32) # 3 times = 56x56
        model = R4Block(in_channel=512)
        assert model(x).shape == torch.Size([1, 1024, 16, 16])

    def test_R5Block(self):
        x = torch.ones(1, 1024, 16, 16) # 3 times = 56x56
        model = R5Block(in_channel=1024)
        assert model(x).shape == torch.Size([1, 2048, 8, 8])


class TestMFN:
    def test_MFN(self):
        model = MFN()
        x = torch.rand(1,3,256,256)
        assert model(x).shape == torch.Size([1,3840,16,16])
    
    def test_FEAT_STRIDE(self):
        assert FEAT_STRIDE == 16
        '''Image Size from 256*256 to 16*16;therefore,FEAT_STRIDE is 265//16 == 16'''

class TestModel:
    model = temp_model()
    x = torch.ones(1,3,200,200)
    output = model(x)

    def test_Temp_part(self):
        assert self.output.shape == torch.Size([1,72,16,16])
