import torch
import torch.nn as nn
class CNNBlock(nn.Module):
    def __init__(self,in_channel,out_channel,act = True,**kwargs):
        super().__init__()
        self.act = act
        self.conv = nn.Conv2d(in_channels = in_channel,
                        out_channels = out_channel,**kwargs)
        self.BN = nn.BatchNorm2d(out_channel)
        self.ReLU = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.BN(x)
        if self.act:
            x = self.ReLU(x)
        return x
class BTNK1(nn.Module):
    def __init__(self,in_channel,half = False,**kwargs):
        super().__init__()

        if half:
            out_channel = in_channel//2
        else:
            out_channel = in_channel
        self.path1 = nn.Sequential(
            CNNBlock(in_channel=in_channel,out_channel=out_channel,kernel_size = 1,**kwargs),
            CNNBlock(in_channel=out_channel,out_channel=out_channel,kernel_size = 3,stride =1,padding =1),
            CNNBlock(in_channel=out_channel,out_channel=4*out_channel,act = False,kernel_size = 1,stride =1)
        )

        self.CNN = CNNBlock(in_channel=in_channel,out_channel=4*out_channel ,act = False,kernel_size = 1,**kwargs)
        self.ReLU = nn.ReLU()
    def forward(self,x):
        x0 = x
        x1 = x
        x0 = self.path1(x0)
        x1 = self.CNN(x1)
        x = x0 + x1
        x = self.ReLU(x)
        return x0
class BTNK2(nn.Module):
    def __init__(self,in_channel,**kwargs):
        super().__init__()
        self.path1 = nn.Sequential(
            CNNBlock(in_channel=in_channel,out_channel=in_channel//4,kernel_size = 1,stride =1),
            CNNBlock(in_channel=in_channel//4,out_channel=in_channel//4,kernel_size = 3,stride =1,padding =1),
            CNNBlock(in_channel=in_channel//4,out_channel=in_channel,act = False,kernel_size = 1,stride =1)
        )
        self.CNN = CNNBlock(in_channel=in_channel,out_channel=in_channel,act = False,kernel_size = 1,**kwargs)
        self.ReLU = nn.ReLU()
    def forward(self,x):
        x0 = x
        x1 = x
        x0 = self.path1(x0)
        x1 = self.CNN(x1)
        x = x0 + x1
        x = self.ReLU(x)
        return x0


class R1Block(nn.Module):
    def __init__(self,in_channel,**kwargs):
        super().__init__()
        self.layers = CNNBlock(in_channel = in_channel,out_channel = 64,kernel_size = 7
                               ,stride =2,padding = 3,**kwargs)
        # self.maxpool = nn.MaxPool2d(kernel_size =3,stride = 2)
    def forward(self,x):
        x = self.layers(x)
        # x = self.maxpool(x)
        return x

class R2Block(nn.Module):
    def __init__(self,in_channel,**kwargs):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size =3,padding =1,stride = 2)
        self.layers = nn.Sequential(
            BTNK1(in_channel),
            BTNK2(in_channel*4),
            BTNK2(in_channel*4)
        )
    def forward(self,x):
        x = self.maxpool(x)
        x = self.layers(x)
        return x

# [1X1 64 --> 3X3 64 --> 1X1 256] 3 times = 56x56
class R3Block(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.Layer_BTNK01 = BTNK1(in_channel,half = True,stride =2)
        self.num_rep =3
        self.Layer_BTNK02 = nn.ModuleList()
        for _ in range(self.num_rep):
            self.Layer_BTNK02+=[
                BTNK2(in_channel*2)
            ]
    def forward(self,x):
        x = self.Layer_BTNK01(x)
        for layer in self.Layer_BTNK02:
            x = layer(x)
        return x


class R4Block(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.Layer_BTNK01 = BTNK1(in_channel,half = True,stride =2)
        self.num_rep =5
        self.Layer_BTNK02 = nn.ModuleList()
        for _ in range(self.num_rep):
            self.Layer_BTNK02+=[
                BTNK2(in_channel*2)
            ]
    def forward(self,x):
        x = self.Layer_BTNK01(x)
        for layer in self.Layer_BTNK02:
            x = layer(x)
        return x

class R5Block(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.Layer_BTNK01 = BTNK1(in_channel,half = True,stride =2)
        self.num_rep =2
        self.Layer_BTNK02 = nn.ModuleList()
        for _ in range(self.num_rep):
            self.Layer_BTNK02+=[
                BTNK2(in_channel*2)
            ]
    def forward(self,x):
        x = self.Layer_BTNK01(x)
        for layer in self.Layer_BTNK02:
            x = layer(x)
        return x