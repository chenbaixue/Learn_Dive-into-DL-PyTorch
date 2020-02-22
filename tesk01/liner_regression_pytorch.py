#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : Nancy
@Time    : 2020/2/14 11:55
@Software: PyCharm
@File    : liner_regression_pytorch.py
@Desc    :使用pytorch的简洁实现
"""
import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
from torch.nn import init
import torch.optim as optim

torch.manual_seed(1)  # 设置随机种子,使实验结果可以复现


def creat_train_data():
    """
    生成样本为1000,特征为2的训练数据
    :return:
    """
    num_inputs = 2
    num_examples = 1000

    # 设定两个影响因素权重和偏差
    true_w = [2, -3.4]
    true_b = 4.2

    # 设置shape和dtype
    features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    # 加上正太分布的随机误差当作噪音
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
    return features, labels


def load_data(features, labels):
    """
    读取数据
    :param features:
    :param labels:
    :return:
    """

    batch_size = 10

    # 将特征和标签组合起来形成数据
    dataset = Data.TensorDataset(features, labels)

    data_iter = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # 批量大小
        shuffle=True,  # 是否顺序混淆
        num_workers=2,  # 读取数据的工作线程数
    )
    return data_iter


class LinearNet(nn.Module):
    """定义模型"""

    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
        # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`

    def forward(self, x):
        y = self.linear(x)
        return y

    """
    这里是定义单层网络,定义多层网络的方法:
    # method one
    net = nn.Sequential(
        nn.Linear(num_inputs, 1)
        # other layers can be added here
        )
    
    # method two
    net = nn.Sequential()
    net.add_module('linear', nn.Linear(num_inputs, 1))
    # net.add_module ......
    
    # method three
    from collections import OrderedDict
    net = nn.Sequential(OrderedDict([
              ('linear', nn.Linear(num_inputs, 1))
              # ......
            ]))
    
    第3个跟前面比较像,区别在于后者有两种好处:
    一是可以自定义隐藏层的名字；
    二是在更深度的情况下支持扩展和方便从外部导入
    """


def train_init(num_inputs):
    net = LinearNet(num_inputs)
    # 初始化模型参数
    init.normal_(net[0].weight, mean=0.0, std=0.01)
    init.constant_(net[0].bias, val=0.0)
    # 定义损失函数
    loss = nn.MSELoss()
    # 定义优化函数
    optimizer = optim.SGD(net.parameters(), lr=0.03)
    # function prototype: `torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    return net, loss, optimizer


def train(data_iter, net, loss, optimizer):
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad()  # 梯度清零
            l.backward()  # 梯度反向传播
            optimizer.step()  # 优化参数
        print('epoch %d, loss: %f' % (epoch, l.item()))
    # 模型结果
    dense = net[0]
    print(dense.weight.data)
    print(dense.bias.data)


if __name__ == '__main__':
    train_features, train_labels = creat_train_data()
    train_data_iter = load_data(train_features, train_labels)
    train_net, train_loss, train_optimizer = train_init(2)
    train(train_data_iter, train_net, train_loss, train_optimizer)
