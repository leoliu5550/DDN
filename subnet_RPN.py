import torch 
import torch.nn as nn
import yaml

class RPN(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        with open('config.yaml','r') as file:
            try:
                self.cfg = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        self.cnn = nn.Conv2d(
            in_channels=in_channel,
            out_channels = 512,
            stride = 1,
            padding = 1,
            kernel_size=3)

                
        # self.nc_score_out = 
        

    def forward(self,x):
        x = self.cnn(x)
        return x