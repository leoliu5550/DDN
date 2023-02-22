import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from Data import *


class Test_RPN_DATA:
    path = 'DATA'
    side_parameter = 3
    data = RPN_DATA(path,side_parameter)
    def test_parameter(self):
        assert self.data.data_cfg['nc'] == 6
        assert self.data.image_path == 'DATA/train/images' 
        assert self.data.label_path == 'DATA/train/labels'
        assert self.data.image_file == ['patches_249.jpg']
        assert len(self.data.image_file) == 1

    def test__len(self):
        assert self.data.__len__() == 2


    def test__getitem(self):
        image_rpn =self.data.__getitem__(0)
        print()
        print(image_rpn[0].shape)
        assert image_rpn[0].shape == torch.Size([3,200,200])
        assert image_rpn[1].shape == torch.Size(
            [6,self.side_parameter,self.side_parameter]
            )
        