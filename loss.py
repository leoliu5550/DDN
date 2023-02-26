import torch.nn as nn
import torch
import yaml
# SLIDE = 16
class RPN_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss = torch.nn.SmoothL1Loss(reduction='none')
        
    def forward(self,pred,target):

        # class obj and noobj loss
		
        loss_obj = torch.zeros(16,16)
        for i in range(0,72,6):
            loss_obj = loss_obj \
                + self.L1_loss(pred[i],target[0]) \
                + self.L1_loss(pred[i+1],target[1]) 


        return None#loss_obj

         
