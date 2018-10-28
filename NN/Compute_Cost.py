# -*- coding: utf-8 -*-
# @Time     : 2018/10/28 17:36
# @Author   : HuangYin
# @FileName : Compute_Cost.py
# @Software : PyCharm
import numpy as np


def compute_Cost(A2, Y, parameter):
    """
       Computes the cross-entropy cost given in equation (13)

       Arguments:
       A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
       Y -- "true" labels vector of shape (1, number of examples)
       parameters -- python dictionary containing your parameters W1, b1, W2 and b2

       Returns:
       cost -- cross-entropy cost given equation (13)
    """
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -np.sum(logprobs) / m

    assert (isinstance(cost, float))

    return cost
