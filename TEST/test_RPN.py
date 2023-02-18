import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from subnet_RPN import *
'''
should preprocess the image size to 2^n
'''

class TestRPN:
    model = RPN(in_channel=3840)
    x = torch.ones(1, 3840,16,16)
    output = model(x)

    def test_RPN_base(self):
        
        assert self.output.shape == torch.Size([1,72,16,16])

    @pytest.mark.skip(reason="just chk cls_pred is ones[16,16]")
    def test_RPN_cls(self):
        print()
        test_ans = torch.zeros(16,16)
        for i in range(24):
            test_ans+=self.output[0][i]
        print(test_ans)
        assert 1==1
        