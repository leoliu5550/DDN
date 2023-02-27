import yaml
import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from loss import *

class TestLoss:
    loss = RPN_loss()
    def test_moment_loss(self):
        pred = torch.ones([1,72,16,16])*2
        target = torch.zeros([1,6,16,16])
        assert self.loss(pred,target) == 9216

# torch.Size([1, 6, 16, 16])
# torch.Size([1, 72, 16, 16])