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
        assert model(x).shape == torch.Size([1, 512, 16, 16])
