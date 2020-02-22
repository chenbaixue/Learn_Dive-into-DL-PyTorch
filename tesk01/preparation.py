#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : Nancy
@Time    : 2020/2/12 17:43
@Software: PyCharm
@File    : preparation.py
pytorch初尝试
"""
import torch
import time


class Timer(object):
    """计时器"""

    def __init__(self):
        self.times = []
        self.start_time = 0
        self.start()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)


if __name__ == '__main__':
    n = 1000
    a = torch.ones(n)
    b = torch.ones(n)

    timer = Timer()
    # 法一:将两个向量使用for循环按元素逐一做标量加法
    c = torch.zeros(n)
    for i in range(n):
        c[i] = a[i] + b[i]
    print('%.5f sec' % timer.stop())

    # 法二:使用torch来将两个向量直接做矢量加法
    timer.start()
    d = a + b
    print('%.5f sec' % timer.stop())
