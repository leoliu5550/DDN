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
        self.L1_loss = torch.nn.SmoothL1Loss(reduction='none')
        self.log_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.base_anchors = AnchorGenerator().generate_anchors()
    def forward(self,pred,target,loc):
        
        print('pred')
        print(pred)
        print('====================================')
        print('target')
        print(target)
        print('====================================')
        print('loc')
        print(loc)
        # pred is {'cls1,cls2,...,box1,box2,...'} 2*12+4*12
        # class obj and noobj loss
        loss_obj = 0
        loss_box = 0
        num_obj_cls = len(anchor_scales)*len(anchor_ratios)
        num_cls_box = len(anchor_scales)*len(anchor_ratios) * 6
        #0~23 cls_loss
        for i in range(0,num_obj_cls*2,2):
            loss_obj = loss_obj\
                + self.log_loss(pred[:,i,...],target[:,0,...]) \
                + self.log_loss(pred[:,i+1,...],target[:,1,...]) 
        loss_obj = loss_obj/(len(anchor_scales)*len(anchor_ratios))


        anchor_index=0
        loc = torch.squeeze(loc)

        for c in range(num_obj_cls*2,num_obj_cls*6,4):# iter in channel
            for x,y in loc:
                x = int(x.item())
                y = int(y.item())
                pred_box = torch.cat((
                    pred[:,c,x,y]/self.base_anchors[anchor_index][0],
                    pred[:,c+1,x,y]/self.base_anchors[anchor_index][1],
                    torch.log(pred[:,c+2,x,y]/self.base_anchors[anchor_index][0]),
                    torch.log(pred[:,c+3,x,y]/self.base_anchors[anchor_index][1])
                ),0)
                



                target_box =torch.cat((
                    target[:,2,x,y]/self.base_anchors[anchor_index][0],
                    target[:,3,x,y]/self.base_anchors[anchor_index][1],
                    torch.log(target[:,4,x,y]/self.base_anchors[anchor_index][0]),
                    torch.log(target[:,5,x,y]/self.base_anchors[anchor_index][1])
                ),0)
            # print('====================================')
            # print('Current C')
            # print(c)
            # print('====================================')
            # print('anchor')
            # print(self.base_anchors[anchor_index][0])
            # print(self.base_anchors[anchor_index][1])
            # print('====================================')
            # print('pred')
            # print(pred_box)
            # print('pred_detail')
            # print(pred[:,c,x,y]/self.base_anchors[anchor_index][0])
            # print(pred[:,c+1,x,y]/self.base_anchors[anchor_index][1])        
            # print(torch.log(pred[:,c+2,x,y]/self.base_anchors[anchor_index][0]))        
            # print(torch.log(pred[:,c+3,x,y]/self.base_anchors[anchor_index][1]))
            # print('====================================')
            # print('target')
            # print(target_box)
            # print('====================================')
            # print('L1_loss')
            # print(self.L1_loss(pred_box,target_box))
            # if torch.isinf(pred).any() or torch.isnan(pred).any():
            #     print("problem")
            #     print(pred)
            anchor_index+=1

        return loss_obj + loss_box

