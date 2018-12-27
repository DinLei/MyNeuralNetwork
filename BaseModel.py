#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/12/26 10:47
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import numpy as np
from collections import Counter


class BaseModel:

    def __init__(self,
                 input_x,
                 input_y):
        self.X = np.array(input_x)
        self.Y = np.array(input_y)

        self.num, self.x_dim = self.X.shape

        assert self.num == self.Y.size

        self.class_counter = Counter(self.Y)
        self.class_num = len(self.class_counter)

        self.parameters = {}
        self.grads = {}

    def _init_parameters(self, hidden_layers_dims):
        hidden_layers_dims = list(hidden_layers_dims)
        hidden_layers_dims.insert(0, self.x_dim)
        hidden_layers_dims.append(self.class_num)
        np.random.seed(1024)
        for i in range(len(hidden_layers_dims)-1):
            self.parameters["W" + str(i)] = np.random.randn(
                hidden_layers_dims[i], hidden_layers_dims[i+1]) * 0.01
            self.parameters["b" + str(i)] = np.zeros(
                (1, hidden_layers_dims[i+1]))


