import yaml
import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from loss import *

class TestLoss:
    loss = RPN_loss()
    # def test_moment_loss(self):
    #     pred = torch.ones([1,72,16,16])*2
    #     target = torch.zeros([1,6,16,16])
    #     assert self.loss(pred,target) == 9216

    def test_value(self):
        x = torch.ones([1,72,2,2]) * 0.2
        y = torch.ones([1,6,2,2]) * 0.7
        self.loss(x,y)

# ANCHOR_SCALES:
#   - 64
#   - 128
#   - 256
#   - 512

# ANCHOR_RATIOS:
#   - 1
#   - 0.5
#   - 2