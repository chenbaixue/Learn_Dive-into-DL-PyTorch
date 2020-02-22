#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : Nancy
@Time    : 2020/2/13 22:46
@Software: PyCharm
@File    : liner_regression_byhand.py
从零实现线性回归模型,预测房价
"""
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random


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


def show_data(features, labels):
    """
    使用图像展示数据
    :param features:
    :param labels:
    :return:
    """
    plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    plt.show()


def data_iter(batch_size, features, labels):
    """
    读取数据集
    :param batch_size:
    :param features:
    :param labels:
    :return:
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 将数据打乱顺序,这样数据不是按顺序排列,因为优化函数采用是是随机梯度下降
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 如果大于batch_size就取i到1000
        yield features.index_select(0, j), labels.index_select(0, j)


def init_model_paramter(num_inputs):
    """
    初始化模型参数
    :return:
    """
    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)

    # 对模型的梯度的附加操作,后面优化函数会用到
    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)
    return w, b


def linreg(X, w, b):
    """
    定义用来训练参数的训练模型
    :param X:
    :param w:
    :param b:
    :return:
    """
    # torch.mm()矩阵相乘
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    """
    定义损失函数
    :param y_hat:
    :param y:
    :return:
    """
    # .view()可以改变tensor的形状
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    定义优化函数，这里是根据梯度下降在梯度的负方向累加一个值来到达优化的效果
    :param params:
    :param lr:
    :param batch_size:
    :return:
    """
    for param in params:
        # 用.data是因为对函数进行优化的这个动作不希望被附加
        param.data -= lr * param.grad / batch_size


def train(batch_size, features, labels, w, b):
    """
    进行模型的训练
    :param labels:
    :param features:
    :param batch_size:小批量样本个数
    :return:
    """
    # 定于超参数
    lr = 0.03  # 训练周期
    num_epochs = 5  # 学习率

    net = linreg  # 单层线性网络
    loss = squared_loss  # 采用常用是均方误差

    # 开始训练
    for epoch in range(num_epochs):
        # 周期训练,每一个周期数据都会被完整的使用一次
        # X是数据, y是标签
        for X, y in data_iter(batch_size, features, labels):
            # 计算损失函数,这里的损失函数是单个样本的,sum()是因为net()得到是batch_size*1的向量,这里需要一个值来表示
            l = loss(net(X, w, b), y).sum()
            l.backward()  # 梯度反向传播
            # 有了梯度就可以进行参数的优化
            sgd([w, b], lr, batch_size)
            # 梯度清零,在pytorch中一个张量的梯度是会不断累加的,如果不清零会影响后面的结果
            w.grad.data.zero_()
            b.grad.data.zero_()
        # 每一个周期结束进行整个批次损失函数的计算,应该是一次减小的
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
        # 模型结果
    print(w, b)


if __name__ == '__main__':
    train_features, train_labels = creat_train_data()
    # show_data(train_features, train_labels)
    w_tensor, b_tensor = init_model_paramter(2)
    train_batch_size = 10
    train(train_batch_size, train_features, train_labels, w_tensor, b_tensor)
