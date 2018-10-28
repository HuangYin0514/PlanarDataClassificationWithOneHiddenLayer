# -*- coding: utf-8 -*-
# @Time     : 2018/10/28 16:52
# @Author   : HuangYin
# @FileName : Forward_Propagation.py
# @Software : PyCharm
import numpy as np
from Sigmoid import sigmoid


def forward_propagation(X, parameters):
    """
       Argument:
       X -- input data of size (n_x, m)
       parameters -- python dictionary containing your parameters (output of initialization function)

       Returns:
       A2 -- The sigmoid output of the second activation
       cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
       """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache
