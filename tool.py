import torch 
# scales = [64,128] 
# ratios = [0.5,1,2]
# FEAT_STRIDE = 16

class AnchorGenerator:
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
    


# gen = AnchorGenerator(scales, ratios)
# print(gen.generate_anchors())
# print(gen.base_anchors)