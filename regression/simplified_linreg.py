import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# 读取数据
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) # PyTorch 中的一个类，用于将多个张量组合成一个数据集。
    return data.DataLoader(dataset, batch_size, shuffle=is_train)# 用于创建一个可迭代的数据加载器，它可以在训练过程中逐批次地提供数据。

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 定义模型
# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))

# 初始化
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# loss
loss = nn.MSELoss()

# 定义优化算哒
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)