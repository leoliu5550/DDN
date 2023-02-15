import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from subnet_RPN import *
'''
should preprocess the image size to 2^n
'''


class TestCNNBlock:
    def test_subnet(self):
        x = torch.ones(1,3840,16,16) # 3 times = 56x56
        model = RPN(in_channel=3840)
        # print(model(x)[0])
        # print(model(x)[1])
        assert model(x)[0].shape == torch.Size([1, 24, 16, 16])
        assert model(x)[1].shape == torch.Size([1, 24, 16, 16])
        assert model(x)[2].shape == torch.Size([1, 48, 16, 16])

