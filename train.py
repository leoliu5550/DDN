import torch 
import torch.nn as nn
import torch.optim as optim
from model import temp_model
from torch.utils.data import DataLoader
from Data import RPN_DATA





# training loop
EPOCH = 10
BATCH = 1

dataloader = DataLoader(dataset=RPN_DATA, batch_size=BATCH, shuffle=False)



for epoch in range(EPOCH):
    for i,(image,labels) in enumerate(dataloader):
        print(i)
        print(image.shape)
        print(labels.shape)



# steps = 1000
# # 定义optim对象
# optimizer = optim.SGD(net.parameters(),lr = 0.01)

# # 在 for循环中更新参数
# for i in range(steps):
#     optimizer.zero_grad() # 将网络中所有的参数的导数都清0
#     output = net(input) # 网络前向传播
#     loss = criterion(output , target) #  计算网络损失
#     loss.backward()# 计算梯度

#     optimizer.step() # 更新参数