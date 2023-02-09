import pytest
import torch
import sys
sys.path.append("/home/lcliu/Documents/DDN")
from Data import *
path = 'DATA'
data = DNN_DATA(path)

class TestDNN_DATA:
    def test_parameter(self):
        assert data.cfg['nc'] == 6
        assert data.image_path == 'DATA/train/images' 
        assert data.label_path == 'DATA/train/labels'

    def test__len(self):
        assert