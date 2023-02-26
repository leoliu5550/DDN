import yaml
import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from loss import *

class TestLoss:
    loss = RPN_loss()
    def test_moment_loss(self):
        

