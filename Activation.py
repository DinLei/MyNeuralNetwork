#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/12/26 16:36
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com


import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_der(x):
    return 1-np.power(tanh(x), 2)


def relu_single(x):
    return max(0, x)


def relu_der_single(x):
    return 0 if x <= 0 else 1


relu = np.vectorize(relu_single)
relu_der = np.vectorize(relu_der_single)


def soft_max(xs, axis=None, keepdims=True):
    num = np.exp(xs)
    return num/np.sum(num, axis=axis, keepdims=keepdims)


def soft_max_der(xs, axis=None, keepdims=True):
    num_ori = np.exp(xs)
    den_ori = np.sum(num_ori, axis=axis, keepdims=keepdims)
    den = np.power(
        np.sum(
            num_ori, axis=axis,
            keepdims=keepdims
        ), 2
    )
    num = num_ori * (den_ori-num_ori)
    return num / den
