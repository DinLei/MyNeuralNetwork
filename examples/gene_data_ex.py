#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/12/27 11:12
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

N = 300  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes

X = np.zeros((N * K, D))
y = np.zeros(N * K, dtype='uint8')
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

nn = NeuralNetwork(X, y)
nn.training([50, 20, 10],
            learning_rate=0.1,
            num_iterations=10000,
            l2_lambda=0.001,
            print_cost=True)
