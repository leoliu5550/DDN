import torch 
import yaml


class AnchorGenerator:
    def __init__(self): #,image_size,feat_stride
        # only need generate Tw and Th
        with open('config.yaml','r') as file:
            cfg = yaml.safe_load(file)
        self.scales = cfg['ANCHOR_SCALES']
        self.ratios = cfg['ANCHOR_RATIOS']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_anchors = torch.zeros(len(self.ratios)*len(self.scales),2)

    def generate_anchors(self):# only need proposal w & h
        scales = torch.as_tensor(self.scales,  device=self.device)
        ratios = torch.as_tensor(self.ratios,  device=self.device)**(1/2)
        ind = 0
        for sca in self.scales:
            for rat in self.ratios:
                
                self.base_anchors[ind][0] = sca/(rat**(1/2)) #as anchor_w
                self.base_anchors[ind][1] = rat*sca/(rat**(1/2)) #as anchor_h
                ind+=1
        return self.base_anchors #contain 12 type of anchor_size


class Indicator:
    @staticmethod
    def IoU(pred,true_ground):
        # assume input size is (1,4,slide,slode)
        batch_num = true_ground.shape[0]
        IoU_loss = 0
        for i in range(batch_num):
            intersection = \
                (true_ground[i,4,...] - (pred[i,2,...]-true_ground[i,2,...]))*(true_ground[i,5,...]-(pred[i,3,...]-true_ground[i,3,...]))
            union = \
                pred[i,4,...]*pred[i,5,...] + true_ground[i,4,...]*true_ground[i,5,...] - intersection
            if torch.equal(union,torch.zeros_like(union)):
                union = 0.00001
        IoU_loss+= torch.sum( intersection/union )
        
        return IoU_loss