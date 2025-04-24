import torch
from d2l import torch as d2l
import random



def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features) # 数据集中样本的总数，通过获取 features 的长度来确定。
    indices = list(range(num_examples)) # 创建一个从 0 到 num_examples - 1 的索引列表。这些索引将用于随机打乱数据。
    random.shuffle(indices) # 使用 random.shuffle 方法随机打乱索引列表。这一步是为了在每次迭代时随机选择样本，从而避免模型在训练过程中对数据的顺序产生依赖。
    for i in range(0, num_examples, batch_size): #  这个循环从 0 开始，每次增加 batch_size，直到 num_examples。i 表示当前批次的起始索引。
        batch_indices = torch.tensor(
            indices[i: min(i+batch_size, num_examples)]) # 从打乱后的索引列表中提取当前批次的索引
        yield features[batch_indices], labels[batch_indices] #  使用提取的索引从 features 和 labels 中获取当前批次的数据。

# 初始化模型
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# 损失函数 loss
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))** 2 /2

# 定义优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
## 随机梯度下降（Stochastic Gradient Descent, SGD）优化算法。

# 训练
# 初始化参数
lr = 0.03 # 学习率，控制参数更新的步长
num_epochs = 3 # 整个数据集被遍历的次数
net = linreg # 模型函数（fuc）
batch_size = 10
loss = squared_loss #损失函数（fuc）

for epoch in range(num_epochs): # 外层循环，每一轮都会遍历整个数据集
    for X, y in data_iter(batch_size, features, labels):# 内层循环，数据读取func
        # X 特征矩阵
        # y标签向量，它们形状分别为(batch_size, features）和（batch——size， 1）
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward() # 反向传播
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():# 禁用梯度计算，因为这里不需要反向传播。
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        # train_l.maen 计算平均损失

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')