import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import io,transforms
import numpy as np
import yaml 
import os 

# PATH = '/data/lcliu/neudet'
class RPN_DATA(Dataset):
    def __init__(self,path,side):
        self.side = side 
        cfg_path = os.path.join(path,"data.yaml")
        print(cfg_path)
        with open(cfg_path, 'r') as file:
            self.data_cfg = yaml.safe_load(file)
        
        self.image_path = os.path.join(path,self.data_cfg['train'][2:])
        self.image_file = os.listdir(self.image_path)
        self.label_path = os.path.join(os.path.split(self.image_path)[0],'labels')
        with open('config.yaml','r') as file:
            self.cfg = yaml.safe_load(file)
            
    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, idx):
        single_image_path = os.path.join(self.image_path,self.image_file[idx])
        
        image = io.read_image(single_image_path).float()
        image = transforms.Resize(256)(image)
        # print('image from DATA.py')
        # print(image.type())
        single_labels_path = os.path.join(self.label_path,self.image_file[idx])
        single_labels_path = os.path.splitext(single_labels_path)[0]+'.txt'
        
        labels = np.loadtxt(fname=single_labels_path, delimiter=" ", ndmin=2)
        RPN_obj= torch.zeros([1,self.side,self.side])
        RPN_nobj = torch.ones([1,self.side,self.side])
        RPN_boxs = torch.zeros([4,self.side,self.side])

        for lines in labels:
            x = int(lines[1] * self.side)
            y = int(lines[2] * self.side)
            RPN_obj[0][x][y] = 1
            RPN_nobj[0][x][y] = 0
            RPN_boxs[0][x][y] = lines[1]-x
            RPN_boxs[1][x][y] = lines[2]-y
            RPN_boxs[2][x][y] = lines[3]
            RPN_boxs[3][x][y] = lines[4]
        RPN_labels = torch.cat((RPN_obj,RPN_nobj,RPN_boxs))

        return image,RPN_labels,single_image_path




# class YOLODataset(Dataset):
#     def __init__(
#         self,
#         csv_file,
#         img_dir,
#         label_dir,
#         anchors,
#         image_size=416,
#         S=[13, 26, 52],
#         C=20,
#         transform=None,
#     ):
#         self.annotations = pd.read_csv(csv_file)
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.image_size = image_size
#         self.transform = transform
#         self.S = S
#         self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
#         self.num_anchors = self.anchors.shape[0]
#         self.num_anchors_per_scale = self.num_anchors // 3
#         self.C = C
#         self.ignore_iou_thresh = 0.5

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
#         bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
#         img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
#         image = np.array(Image.open(img_path).convert("RGB"))

#         if self.transform:
#             augmentations = self.transform(image=image, bboxes=bboxes)
#             image = augmentations["image"]
#             bboxes = augmentations["bboxes"]

#         # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
#         targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
#         for box in bboxes:
#             iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
#             anchor_indices = iou_anchors.argsort(descending=True, dim=0)
#             x, y, width, height, class_label = box
#             has_anchor = [False] * 3  # each scale should have one anchor
#             for anchor_idx in anchor_indices:
#                 scale_idx = anchor_idx // self.num_anchors_per_scale
#                 anchor_on_scale = anchor_idx % self.num_anchors_per_scale
#                 S = self.S[scale_idx]
#                 i, j = int(S * y), int(S * x)  # which cell
#                 anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
#                 if not anchor_taken and not has_anchor[scale_idx]:
#                     targets[scale_idx][anchor_on_scale, i, j, 0] = 1
#                     x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
#                     width_cell, height_cell = (
#                         width * S,
#                         height * S,
#                     )  # can be greater than 1 since it's relative to cell
#                     box_coordinates = torch.tensor(
#                         [x_cell, y_cell, width_cell, height_cell]
#                     )
#                     targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
#                     targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
#                     has_anchor[scale_idx] = True

#                 elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
#                     targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

#         return image, tuple(targets)


