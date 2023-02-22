import torch
import torch.nn as nn
import torch.nn.functional as f
from subnet_RPN import RPN
IMAGE_SIZE = 200
FEAT_STRIDE = 16

class CNNBlock(nn.Module):
    def __init__(self,in_channel,out_channel,act = True,**kwargs):
        super().__init__()
        self.act = act
        self.conv = nn.Conv2d(in_channels = in_channel,
                        out_channels = out_channel
                        ,**kwargs)
        self.bn = nn.BatchNorm2d(out_channel)
        self.ReLU = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
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
        x0 = self.path1(x0)
        x1 = x
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
        x0 = self.path1(x0)
        x1 = x
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

class MFN(nn.Module):
    def __init__(self):
        super().__init__()
        in_list = [3,64,256,512,1024,2048] #in_channel in R1 R2 R3 R4 R5 

        self.r1 = R1Block(in_list[0])
        self.r2 = R2Block(in_list[1])

        self.r2Conv = nn.Sequential(
            nn.Conv2d(in_channels = in_list[2],out_channels = in_list[2],kernel_size=3,stride=2,padding =1),
            nn.Conv2d(in_channels = in_list[2],out_channels = in_list[2],kernel_size=3,stride=2,padding =1),
            nn.Conv2d(in_channels = in_list[2],out_channels = in_list[2],kernel_size=1,stride=1)
        )

        self.r3 = R3Block(in_list[2])
        self.r3Pool = nn.MaxPool2d(kernel_size = 2,stride=2)
        self.r4 = R4Block(in_list[3])
        self.r4Conv = nn.Conv2d(in_channels = in_list[4],out_channels = in_list[4],kernel_size=1,stride=1)
        self.r5 = R5Block(in_list[4])
        self.r5Conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_list[5],out_channels = in_list[5],kernel_size = 4,stride=2,padding =1),
            nn.Conv2d(in_channels = in_list[5],out_channels = in_list[5],kernel_size=1,stride=1)
        )

    def forward(self,x):
        R_output= []
        x = self.r1(x)
        x = self.r2(x)
        R_output.append(x) # R_output+= x
        R_output[-1] = self.r2Conv(R_output[-1])

        x = self.r3(x)
        R_output.append(x) # R_output+= x
        R_output[-1] = self.r3Pool(R_output[-1])
        x = self.r4(x)
        R_output.append(x) # R_output+= x
        R_output[-1] = self.r4Conv(R_output[-1])
        x = self.r5(x)
        R_output.append(x) # R_output+= x
        R_output[-1] = self.r5Conv(R_output[-1])
        for i,out in enumerate(R_output):
            print(R_output[i].shape)
            R_output[i] = f.normalize(out,p=2) #,dim=-1

        output = torch.cat(R_output,dim=1)

        # global  FEAT_STRIDE
        # FEAT_STRIDE = x.shape[2] // output.shape[2]
        return output

class temp_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.MFN = MFN()
        self.RPN = RPN()
    
    def forward(self,x):
        x = MFN(x)
        x = RPN(x)

        return x
