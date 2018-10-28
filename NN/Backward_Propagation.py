# -*- coding: utf-8 -*-
# @Time     : 2018/10/28 18:44
# @Author   : HuangYin
# @FileName : Backward_Propagation.py
# @Software : PyCharm
import numpy as np


def backward_propagation(parameters, cache, X, Y):
    """
       Implement the backward propagation using the instructions above.

       Arguments:
       parameters -- python dictionary containing our parameters
       cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
       X -- input data of shape (2, number of examples)
       Y -- "true" labels vector of shape (1, number of examples)

       Returns:
       grads -- python dictionary containing your gradients with respect to different parameters
       """
    m = X.shape[1]

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW2": dW2,
        "dW1": dW1,
        "db2": db2,
        "db1": db1

    }
    return grads
