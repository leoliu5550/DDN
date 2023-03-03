import torch.nn as nn
import torch
import yaml
from tool import Indicator,AnchorGenerator
# SLIDE = 16
with open('config.yaml','r') as file:
    try:
        _cfg = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
anchor_ratios = _cfg['ANCHOR_RATIOS']
anchor_scales = _cfg['ANCHOR_SCALES']

class RPN_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss = torch.nn.SmoothL1Loss(reduction='sum')
        self.log_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.base_anchors = AnchorGenerator().generate_anchors()
    def forward(self,pred,target):
        # pred is {'cls1,cls2,...,box1,box2,...'} 2*12+4*12
        # class obj and noobj loss
        loss_obj = 0
        loss_box = 0
        num_obj_cls = len(anchor_scales)*len(anchor_ratios)
        num_cls_box = len(anchor_scales)*len(anchor_ratios) * 6
        for i in range(num_obj_cls,num_obj_cls*2,2):
            loss_obj = loss_obj\
                + self.log_loss(pred[:,i,...],target[:,0,...]) \
                + self.log_loss(pred[:,i+1,...],target[:,1,...]) 
        loss_obj = loss_obj/(len(anchor_scales)*len(anchor_ratios))
        
        for i in range(num_obj_cls,num_cls_box ,4):
            pass




        return loss_obj + loss_box

