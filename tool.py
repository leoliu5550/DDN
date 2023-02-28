import torch 
import yaml


class Generator:
    def __init__(self,scales,ratios):
        self.scales = scales
        self.ratios = ratios
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_anchors = torch.zeros([len(self.ratios)*len(self.scales),4  ])
        pass

    def generate_anchors(self):
        '''
        give the base anchor at cell center as (0,0)
        '''
        scales = torch.as_tensor(self.scales,  device=self.device)
        ratios = torch.as_tensor(self.ratios,  device=self.device)
        h_ratios = ratios
        anchors_h = (h_ratios[:,None] * scales[None,:]).view(-1)
        w_ratios  = 1.0/ratios
        anchors_w = (w_ratios[:,None] * scales[None,:]).view(-1)

        self.base_anchors = torch.stack([ anchors_w, anchors_h], dim=1)
        
        return self.base_anchors # change only return w and h
    
# torch.Size([12, 4])
# tensor([[ -64.,   16.,   64.,  -16.],
#         [-128.,   32.,  128.,  -32.],
#         [-256.,   64.,  256.,  -64.],
#         [-512.,  128.,  512., -128.],
#         [ -32.,   32.,   32.,  -32.],
#         [ -64.,   64.,   64.,  -64.],
#         [-128.,  128.,  128., -128.],
#         [-256.,  256.,  256., -256.],
#         [ -16.,   64.,   16.,  -64.],
#         [ -32.,  128.,   32., -128.],
#         [ -64.,  256.,   64., -256.],
#         [-128.,  512.,  128., -512.]], device='cuda:0')

class AnchorGenerator(Generator):
    def __init__(self, scales, ratios): #,image_size,feat_stride
        super().__init__(scales, ratios)
        # self.feature_szie = image_size/feat_stride
        self.base_anchors = self.generate_anchors()

    def RPN_proposal_target(self):
        return None

    @staticmethod # should fixe
    def get_LabelInfo(label_path):
        with open(label_path,'r') as file:
            # x0,y0,w,h
            cfg = file.read().splitlines()
        data = []
        for lines in cfg:
            ln = lines.split(' ')
            ln = [eval(i) for i in ln]
            data.append([ln[0],ln[1],ln[2],ln[1]+ln[3],ln[2]+ln[4]])
        # x = xmin
        # y = ymin
        # w = xmax - xmin
        # h = ymax - ymin
        return data


class Indicator:
    @staticmethod
    def IoU(pred,true_ground):
        
        intersection = \
            true_ground[2,...] * true_ground[3,...].transpose()# \
            # - (true_ground[2]*(pred[1]-true_ground[1]) + true_ground[3]*(pred[0]-true_ground[0])) \
            # + (pred[0]-true_ground[0])*(pred[1]-true_ground[1])

        # union = \
        #     pred[2] * pred[3] \
        #     + (true_ground[2]*(pred[1]-true_ground[1]) + true_ground[3]*(pred[0]-true_ground[0])) \
        #     - (pred[0]-true_ground[0])*(pred[1]-true_ground[1])
    
        return intersection #/union