# -*- coding: utf-8 -*-
# @Time     : 2018/10/28 16:58
# @Author   : HuangYin
# @FileName : Sigmoid.py
# @Software : PyCharm
import numpy as np

def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a