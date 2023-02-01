import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from model import *

class TestMFN:
    def test_MFN(self):
        model = MFN()
        x = torch.rand(1,3,320,320)
        assert model(x).shape == torch.Size([1,3840,20,20])

