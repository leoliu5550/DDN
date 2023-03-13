import torch 
import torch.nn as nn
import yaml
from tool import AnchorGenerator

class RPN(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        with open('config.yaml','r') as file:
            try:
                _cfg = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        # get all init anchor x0,y0,x1,y1
        self.base_anchors = AnchorGenerator().generate_anchors()
        # 圖像輸入RPN
        self.cnn = nn.Conv2d(
                        in_channels=in_channel,
                        out_channels = 512,
                        stride = 1,
                        padding = 1,
                        kernel_size=3)

        self._training = _cfg['TRAINING']
        self.anchor_ratios = _cfg['ANCHOR_RATIOS']
        self.anchor_scales = _cfg['ANCHOR_SCALES']
        # 產生出 len(self.anchor_ratios) * len(self.anchor_scales)種先驗框 ＊２背景or not
        self.nc_score_out = len(self.anchor_ratios) * len(self.anchor_scales) * 2
        self.rpn_objcls_pred = nn.Conv2d(
            in_channels = 512,
            out_channels = self.nc_score_out,
            kernel_size = 1,
            stride = 1,
            padding = 0
            )

        #產生出 self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 x,y,w,h
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.rpn_box_pred = nn.Conv2d(
            in_channels = 512,
            out_channels = self.nc_bbox_out,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        

        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        

        
    def forward(self,x):
        x = self.cnn(x)
        x = self.ReLU(x)
        box_pred = self.rpn_box_pred(x)
        box_pred = self.ReLU(box_pred)
        anchor_index=0
        for c in range(0,self.nc_bbox_out,4):
            box_pred[:,c,...] = box_pred[:,c,...] * self.base_anchors[anchor_index][0]
            box_pred[:,c+1,...] = box_pred[:,c+1,...] * self.base_anchors[anchor_index][1]
            box_pred[:,c+2,...] = self.base_anchors[anchor_index][0] * torch.exp(box_pred[:,c+2,...])
            box_pred[:,c+3,...] = self.base_anchors[anchor_index][1] * torch.exp(box_pred[:,c+3,...])
            anchor_index+=1
        # channel is 24 per 2
        cls_pred = self.rpn_objcls_pred(x).squeeze(0).permute(1, 2, 0).contiguous()
        cls_pred = cls_pred.view(-1, 2)
        cls_pred = self.softmax(self.rpn_objcls_pred(x))
        output = torch.cat((cls_pred,box_pred),dim=1)
        # output = [1,24,16,16] cat [1,48,16,16]

        return output

