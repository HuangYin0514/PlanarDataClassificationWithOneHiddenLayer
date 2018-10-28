# -*- coding: utf-8 -*-
# @Time     : 2018/10/28 20:54
# @Author   : HuangYin
# @FileName : Predict.py
# @Software : PyCharm
import numpy as np
from Forward_Propagation import forward_propagation


def predict(parameter, X):
    A2, cache = forward_propagation(X, parameter)
    predictions = np.round(A2)
    return predictions
