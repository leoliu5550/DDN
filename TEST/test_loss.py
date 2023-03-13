import yaml
import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from loss import *

class TestLoss:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    loss_fn = RPN_loss()
    pred = torch.load("./pred.pt").to(device)
    ans  = torch.load("./ans.pt").to(device)
    loc  = torch.load("./loc.pt").to(device)


    def test_valueState(self):
        loss = self.loss_fn(self.pred,self.ans,self.loc)
        print()
        print(loss[0])
        print(loss[1])
        # assert loss.loss_24inex() == False
        # assert loss.loss_68inex == False