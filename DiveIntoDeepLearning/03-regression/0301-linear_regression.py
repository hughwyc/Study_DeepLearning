# y = w*x +b,
# 给定训练数据特征x和对应的已知标签y，线性回归的目标是找到一组权重向量w和偏置b：
# 当给定从x的同分布中取样的新样本特征时，这组权重向量和偏置能够使得新样本预测标签的误差尽可能小。

# 损失函数（loss function）能够量化目标的实际值与预测值之间的差距。

# 梯度下降最简单的用法是计算损失函数（数据集中所有样本的损失均值） 关于模型参数的导数（在这里也可以称为梯度）。
# 但实际中的执行可能会非常慢：因为在每一次更新参数之前，我们必须遍历整个数据集。
# 因此，我们通常会在每次需要计算更新的时候随机抽取一小批样本，
# 这种变体叫做小批量随机梯度下降（minibatch stochastic gradient descent）。

# 线性回归恰好是一个在整个域中只有一个最小值的学习问题。
# 但是对像深度神经网络这样复杂的模型来说，损失平面上通常包含多个最小值。
# 深度学习实践者很少会去花费大力气寻找这样一组参数，使得在训练集上的损失达到最小。
# 事实上，更难做到的是找到一组参数，这组参数能够在我们从未见过的数据上实现较低的损失， 这一挑战被称为泛化（generalization）。

import math
import numpy as np
import torch

import sys

sys.path.append("..")
from utils.tools import Timer

n = 10000
a = torch.ones([n])
b = torch.ones([n])

c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')

timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')
