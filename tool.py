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
        # assume input size is (1,6,slide,slode)


        Uperx = torch.max(pred[:,0,...] - pred[:,2,...]/2,\
                        true_ground[:,0,...] - true_ground[:,2,...]/2)
        Upery = torch.max(pred[:,1,...] - pred[:,3,...]/2,\
                        true_ground[:,1,...] - true_ground[:,3,...]/2)
        
        Botx = torch.min(pred[:,0,...] - pred[:,2,...]/2,\
                        true_ground[:,0,...] - true_ground[:,2,...]/2)
        
        Boty = torch.min(pred[:,1,...] - pred[:,3,...]/2,\
                        true_ground[:,1,...] - true_ground[:,3,...]/2)

        intersection = torch.abs((Uperx - Botx) * (Upery - Boty))
        union = torch.abs(pred[:,4,...] * pred[:,5,...] \
            + true_ground[:,4,...] * true_ground[:,5,...])

        
        if torch.equal(union,torch.zeros_like(union)):
            union = 0.00001
        IoU_inde = intersection/union
        for i in range(IoU_inde.shape[1]):
            for j in range(IoU_inde.shape[2]):
                if IoU_inde[0][i][j]>0.7 or IoU_inde[0][i][j]<0.3:
                    IoU_inde[0][i][j] = 1
                else:
                    IoU_inde[0][i][j] = 0

        return IoU_inde
