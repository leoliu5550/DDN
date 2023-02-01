import torch
import torch.nn as nn
import torch.nn.functional as f
IMAGE_SIZE = 200
class CNNBlock(nn.Module):
    def __init__(self,in_channel,out_channel,act = True,**kwargs):
        super().__init__()
        self.act = act
        self.conv = nn.Conv2d(in_channels = in_channel,
                        out_channels = out_channel,**kwargs)
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

class MFN(nn.Module):
    def __init__(self):
        super().__init__()
        in_list = [3,64,256,512,1024,2048]
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
            R_output[i] = f.normalize(out,p=2) #,dim=-1
        output = torch.cat(R_output,dim=1)
        print(output.shape)
        return output


class RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super().__init__()
        
        self.din = din  # 特征图的深度, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES	# 锚框的尺度变化量，默认是[8, 16 ,32]
        self.anchor_ratios = cfg.ANCHOR_RATIOS  # 锚框的宽高变化量,默认是[0.5, 1 ,2]
        self.feat_stride = cfg.FEAT_STRIDE[0]   # 特征图的下采样倍数，默认是16

        # 定义conv层处理输入特征映射
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # 定义前景/背景分类层
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # 定义锚框偏移量预测层
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # 定义区域生成模块
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # 定义生成RPN训练标签模块，仅在训练时使用
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

		# RPN分类损失以及回归损失，仅在训练时计算
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
    	# 用于修改张量形状
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
 
        # 输入数据的第一维是batch数
        batch_size = base_feat.size(0)

        # 首先利用3×3卷积进一步融合特征
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # 利用1×1卷积得到分类网络，每个点代表anchor的前景背景得分
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        # 利用reshape与softmax得到anchor的前景背景概率
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # 利用1×1卷积得到回归网络，每一个点代表anchor的偏移
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # 区域生成模块
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # 生成RPN训练时的标签和计算RPN的loss
        if self.training:
            assert gt_boxes is not None

			# 生成训练标签
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # 计算分类损失
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = rpn_label.view(-1).ne(-1).nonzero().view(-1)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = rpn_label.long()
            # 先对scores进行筛选得到256个样本的得分，随后进行交叉熵求解
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # 利用smoothl1损失函数进行loss计算
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box
