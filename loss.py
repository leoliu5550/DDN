import torch.nn as nn
import torch
import yaml

class RPN_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss = torch.nn.SmoothL1Loss()
        
    def forward(self,pred,target,anchor_locate):
        pred[] 

        loss = self.L1_loss(pred,target)
        return loss

         