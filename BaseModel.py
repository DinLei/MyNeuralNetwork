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

    def _init_parameters(self, hidden_layers_dims, initialization="he"):
        layers_dims = [self.x_dim]
        layers_dims.extend(hidden_layers_dims)
        layers_dims.append(self.class_num)
        if initialization == "he":
            self.parameters = self._init_parameters_he(layers_dims)
        elif initialization == "zero":
            self.parameters = self._init_parameters_random(layers_dims, 0.0)
        elif initialization == "random":
            self.parameters = self._init_parameters_random(layers_dims, 0.01)

    @staticmethod
    def _init_parameters_random(layers_dims, base):
        np.random.seed(1024)
        parameters = {}
        for i in range(len(layers_dims)-1):
            parameters["W" + str(i)] = np.random.randn(
                layers_dims[i], layers_dims[i+1]) * base
            parameters["b" + str(i)] = np.zeros((1, layers_dims[i+1]))
        return parameters

    @staticmethod
    def _init_parameters_he(layers_dims):
        np.random.seed(1024)
        parameters = {}
        for l in range(len(layers_dims) - 1):
            parameters['W' + str(l)] = np.random.randn(
                layers_dims[l],
                layers_dims[l+1]
            ) * np.sqrt(2 / layers_dims[l])
            parameters['b' + str(l)] = np.zeros((1, layers_dims[l+1]))
        return parameters
