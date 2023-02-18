import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from Data import *
path = 'DATA'
data = RPN_DATA(path,5)

class TestDNN_DATA:
    def test_parameter(self):
        assert data.data_cfg['nc'] == 6
        assert data.image_path == 'DATA/train/images' 
        assert data.label_path == 'DATA/train/labels'
        assert data.image_file == ['patches_249.jpg']
        assert len(data.image_file) == 1

    def test__len(self):
        assert data.__len__() == 1


    def test__getitem(self):
        image_rpn =data.__getitem__(0)
        print(image_rpn)
        assert image_rpn.shape == torch.Size([6,5,5])
