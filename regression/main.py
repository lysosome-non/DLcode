import math
import numpy as np
import torch
from d2l import torch as d2l
import random

from d2l.tensorflow import numpy


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
