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

        self.base_anchors = torch.stack([ -anchors_w, anchors_h, anchors_w, -anchors_h], dim=1)/ 2
        
        return self.base_anchors
    
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






# scales = [64,128,256,512] 
# ratios = [0.5,1,2]
# FEAT_STRIDE = 16

# anc = AnchorGenerator(scales,ratios)#,feat_stride=FEAT_STRIDE)
# print(anc.get_LabelInfo('sample/patches_249.txt'))



# Draw Image
# from PIL import Image ,ImageDraw
# from tool import AnchorGenerator

# scales = [64,128,256,512] 
# ratios = [0.5,1,2]
# FEAT_STRIDE = 16

# anc = AnchorGenerator(scales,ratios)#,feat_stride=FEAT_STRIDE)
# label = anc.get_LabelInfo('sample/patches_249.txt') 
# print(label)
# img = Image.open(r"sample/patches_249.jpg")
# a = ImageDraw.Draw(img)
# for i in label:
#     i = [l * 200 for l in i]
#     print(i)
#     a.rectangle(((i[1],i[2]),(i[3],i[4])),outline='red',width = 3)
# img.save("sample/test.jpg")