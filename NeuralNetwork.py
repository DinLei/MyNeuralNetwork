#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/12/26 10:45
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

from Activation import *
from BaseModel import BaseModel
import matplotlib.pyplot as plt


class NeuralNetwork(BaseModel):
    """
    shape of input in every layer is (num, x_{i}_dim)
    shape of current W is (x_{i}_dim, x_{i+1}_dim)
    shape of current z is (num, x_{i+1}_dim)
    here the code need hidden layers dims, this obj is a list
    but not include x_dim and output_dim, these two dims will
    auto add into the list when parameters inited by check the
    data uploaded
    """
    def __init__(self,
                 input_x,
                 input_y):
        BaseModel.__init__(self,
                           input_x,
                           input_y)
        self.l2_lambda = None

    def _get_l2_loss(self):
        l2_loss = 0
        if self.parameters and self.l2_lambda:
            layers = len(self.parameters) // 2
            for i in range(layers):
                curr_w = self.parameters["W"+str(i)]
                l2_loss += np.sum(curr_w*curr_w)
            l2_loss *= (0.5 * self.l2_lambda)
        return l2_loss

    def _get_l2_reg_gradient(self, i):
        if self.parameters and self.l2_lambda:
            return self.l2_lambda * self.parameters["W"+str(i)]
        return None

    def _model_forward(self, input_x, keep_prob=1):
        caches = []
        layers = len(self.parameters) // 2
        a_i = input_x
        for i in range(layers-1):
            a_i_prev = a_i
            z_i, linear_cache = linear_forward(a_i_prev,
                                               self.parameters["W"+str(i)],
                                               self.parameters["b"+str(i)])
            a_i, activation_cache = activate_forward(z_i, "relu", keep_prob=keep_prob)
            caches.append((linear_cache, activation_cache))
        z_l, linear_cache = linear_forward(a_i,
                                           self.parameters["W" + str(layers-1)],
                                           self.parameters["b" + str(layers-1)])
        a_l, activation_cache = activate_forward(z_l, "soft_max", axis=1, keep_prob=1)
        caches.append((linear_cache, activation_cache))

        assert a_l.shape == (input_x.shape[0], self.class_num)
        return a_l, caches

    def _model_backward(self, a_l, caches, input_y):
        ele_num, class_num = a_l.shape
        da_l = a_l
        da_l[range(ele_num), input_y] -= 1
        layers = len(caches)
        curr_cache = caches[-1]
        linear_cache, activation_cache = curr_cache
        dz_l = activation_backward(da_l, activation_cache, "soft_max", axis=1)
        da_prev, dw_l, db_l = linear_backward(dz_l,
                                              linear_cache,
                                              self._get_l2_reg_gradient(layers-1))
        self.grads["da"+str(layers-1)] = da_prev
        self.grads["dW"+str(layers-1)] = dw_l
        self.grads["db"+str(layers-1)] = db_l

        for j in reversed(range(layers-1)):
            curr_cache = caches[j]
            linear_cache, activation_cache = curr_cache
            da_j = keep_prob_backward(da_prev=da_prev, activation_cache=activation_cache)

            dz_j = activation_backward(da_j, activation_cache, "relu")
            da_prev, dw_j, db_j = linear_backward(dz_j,
                                                  linear_cache,
                                                  self._get_l2_reg_gradient(j))
            self.grads["da" + str(j)] = da_prev
            self.grads["dW" + str(j)] = dw_j
            self.grads["db" + str(j)] = db_j

    def _update_parameters(self, learning_rate):
        layers = len(self.parameters) // 2
        for i in range(layers):
            self.parameters["W"+str(i)] -= learning_rate * self.grads["dW"+str(i)]
            self.parameters["b" + str(i)] -= learning_rate * self.grads["db" + str(i)]

    def _update_parameters_with_adam(self,
                                     t=3,
                                     learning_rate=0.01,
                                     beta1=0.9, beta2=0.999,
                                     epsilon=1e-8):

        layers = len(self.parameters) // 2

        v_corrected = {}
        s_corrected = {}

        for l in range(layers):
            self.velocity["dW" + str(l)] = \
                beta1 * self.velocity["dW" + str(l)] + \
                (1 - beta1) * self.grads['dW' + str(l)]
            self.velocity["db" + str(l)] = \
                beta1 * self.velocity["db" + str(l)] + \
                (1 - beta1) * self.grads['db' + str(l)]

            v_corrected["dW" + str(l)] = self.velocity["dW" + str(l)] / (1 - np.power(beta1, t))
            v_corrected["db" + str(l)] = self.velocity["db" + str(l)] / (1 - np.power(beta1, t))

            self.rms_prop["dW" + str(l)] =\
                beta2 * self.rms_prop["dW" + str(l)] + \
                (1 - beta2) * np.power(self.grads['dW' + str(l)], 2)
            self.rms_prop["db" + str(l)] = \
                beta2 * self.rms_prop["db" + str(l)] + \
                (1 - beta2) * np.power(self.grads['db' + str(l)], 2)

            s_corrected["dW" + str(l)] = self.rms_prop["dW" + str(l)] / (1 - np.power(beta2, t))
            s_corrected["db" + str(l)] = self.rms_prop["db" + str(l)] / (1 - np.power(beta2, t))

            self.parameters["W" + str(l)] = \
                self.parameters["W" + str(l)] - \
                learning_rate * v_corrected["dW" + str(l)] / np.sqrt(s_corrected["dW" + str(l)] + epsilon)
            self.parameters["b" + str(l)] = \
                self.parameters["b" + str(l)] - \
                learning_rate * v_corrected["db" + str(l)] / np.sqrt(s_corrected["db" + str(l)] + epsilon)

    def training(self,
                 layers_dims,
                 learning_rate=0.075,
                 num_iterations=1000,
                 batch_size=100,
                 l2_lambda=None,
                 keep_prob=1,
                 optimizer="adam",
                 print_cost=False):
        costs = []
        self.l2_lambda = l2_lambda
        self._init_parameters(hidden_layers_dims=layers_dims)
        for i in range(num_iterations):
            index = list(range(self.num))
            np.random.shuffle(index)
            if not batch_size:
                batch_size = self.num
            piece = self.num // batch_size
            for j in range(piece):
                if j == piece - 1:
                    part = tuple(index[j * batch_size:])
                else:
                    part = tuple(index[j * batch_size: (j + 1) * batch_size])
                batch_x = self.X[part, :]
                batch_y = self.Y[list(part)]
                a_l, caches = self._model_forward(batch_x, keep_prob=keep_prob)
                cost = cross_entropy_loss(a_l, batch_y, self._get_l2_loss())
                self._model_backward(a_l, caches, batch_y)
                if optimizer == "adam":
                    self._update_parameters_with_adam(learning_rate=learning_rate)
                else:
                    self._update_parameters(learning_rate)
            if print_cost and i % 200 == 0:
                print("Cost after iteration {i}: {c}".format(i=i, c=cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()


def linear_forward(a_i, w_i, b_i):
    z_i = np.dot(a_i, w_i) + b_i
    assert z_i.shape == (a_i.shape[0], w_i.shape[1])
    cache = (a_i, w_i, b_i)
    return z_i, cache


def activate_forward(z_i, activation, axis=None, keepdims=True, keep_prob=1):
    assert 0 < keep_prob <= 1
    if activation == "sigmoid":
        a_i1 = sigmoid(z_i)
    elif activation == "tanh":
        a_i1 = tanh(z_i)
    elif activation == "relu":
        a_i1 = relu(z_i)
    elif activation == "soft_max":
        a_i1 = soft_max(z_i, axis=axis, keepdims=keepdims)
    else:
        raise Exception("Have not def this activation function!")
    nrow, ncol = a_i1.shape
    drop_mask = np.random.rand(nrow, ncol) < keep_prob
    a_i1 *= drop_mask
    a_i1 = a_i1 / keep_prob
    cache = (z_i, drop_mask, keep_prob)
    return a_i1, cache


def cross_entropy_loss(prob, y, l2_loss=None):
    """
    In my code, prob is a_l
    :param prob: 
    :param y: 
    :param l2_loss: 
    :return: 
    """
    ele_num, _ = prob.shape
    assert ele_num == len(y) and ele_num >= 1
    correct_loss = -np.log(prob[range(ele_num), y])
    cost = np.sum(correct_loss) / ele_num
    if l2_loss:
        cost += l2_loss
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())
    return cost


def linear_backward(dz_i, linear_cache, l2_reg_gradient=None):
    """
    In this layer, shape of a_i_prev is (num, a_{i-1}_dim) and the shape of dz_i is (num, a_{i}_dim).
    So the shape of w and dw is (a_{i-1}_dim, a_{i}_dim)
    :param dz_i: 
    :param linear_cache: 
    :param l2_reg_gradient: 
    :return: 
    """
    ele_num = len(dz_i)
    a_i_prev, w_i, b_i = linear_cache
    assert a_i_prev.shape[0] == ele_num
    dw_i = np.dot(a_i_prev.T, dz_i)/ele_num
    if l2_reg_gradient is not None:
        dw_i += l2_reg_gradient
    db_i = np.squeeze(np.sum(dz_i, axis=0, keepdims=True))/ele_num
    db_i = db_i.reshape((1, db_i.size))

    da_prev = np.dot(dz_i, w_i.T)

    return da_prev, dw_i, db_i


def activation_backward(da_i, activation_cache, activation, axis=None, keepdims=True):
    """
    :param da_i: shape of a_i is (num, a_{i}_dim)
    :param activation_cache: shape of z_i is (num, z_{i}_dim), a_i = g(z_i)
    :param activation: 
    :param axis: 
    :param keepdims: 
    :return: 
    """
    z_i, _, _ = activation_cache

    if activation == "sigmoid":
        gz_i = sigmoid_der(z_i)
    elif activation == "tanh":
        gz_i = tanh_der(z_i)
    elif activation == "relu":
        gz_i = relu_der(z_i)
    elif activation == "soft_max":
        gz_i = soft_max_der(z_i, axis=axis, keepdims=keepdims)
    else:
        raise Exception("Have not def this activation function!")
    return da_i * gz_i


def keep_prob_backward(da_prev, activation_cache):
    if activation_cache:
        _, drop_mask, keep_prob = activation_cache
        da_prev *= drop_mask
        da_prev = da_prev / keep_prob
    return da_prev

