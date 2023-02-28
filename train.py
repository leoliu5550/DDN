import yaml
import os
import torch 
import torch.nn as nn
import torch.optim as optim
from model import RPN_model
from torch.utils.data import DataLoader
from Data import RPN_DATA
from loss import RPN_loss

with open('config.yaml','r') as file:
    _cfg = yaml.safe_load(file)


MODEL_PATH = os.path.join('model',_cfg['MODELNAME'][0])

# training loop
EPOCH = 100000 # problem
BATCH = 1
SLIDE = 16
LEARNING_RATE = 0.001

dataloader = DataLoader(
    dataset=RPN_DATA('DATA',SLIDE),
    batch_size=BATCH,
    shuffle=False)



# GPU config
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

if  os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH)
else:
    model = RPN_model()
model=model.to(device)




loss_fn = RPN_loss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

for epoch in range(EPOCH):
    for i, (image,labels,picture_name) in enumerate(dataloader):
        
        optimizer.zero_grad()
        image = image.to(device)
        labels = labels.to(device)

        outputs = model(image)
        loss = loss_fn(outputs,labels)
        print(i,picture_name,loss)
        loss.backward()
        optimizer.step()

torch.save(model,MODEL_PATH)




# from torchstat import stat
# stat(model, (3, 224, 224))

