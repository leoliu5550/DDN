from torchstat import stat
import torch 

FILE = 'model/tempmodel.pt'

model = torch.load(FILE)



stat(model, (3, 224, 224))

