# -*- coding: utf-8 -*-
# @Time     : 2018/10/28 16:33
# @Author   : HuangYin
# @FileName : Layer_sizes.py
# @Software : PyCharm

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)