import torch 
import torch.nn as nn
import torch.optim as optim
from model import RPN_model
from torch.utils.data import DataLoader

from Data import RPN_DATA
from loss import RPN_loss




# training loop
EPOCH = 1
BATCH = 1
SLIDE = 16
LEARNING_RATE = 0.001

dataloader = DataLoader(
    dataset=RPN_DATA('DATA',SLIDE),
    batch_size=BATCH,
    shuffle=False)
dataiter = iter(dataloader)

model = RPN_model()
loss_fn = RPN_loss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

for epoch in range(EPOCH):
    image,labels,picture_name = next(dataiter)
    print(picture_name)
    print('image')
    print(image.type())
    print(image.shape)
    print('labels')
    print(labels.type())
    print(labels.shape)
    optimizer.zero_grad()
    outputs = model(image)
    loss = loss_fn(outputs,labels)
    # loss.backward()
    # optimizer.step()
