# -*- coding: utf-8 -*-
# @Time     : 2018/10/29 9:15
# @Author   : HuangYin
# @FileName : MoreModel.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset, plot_decision_boundary
import sklearn
from testCases import layer_sizes_test_case, initialize_parameters_test_case, forward_propagation_test_case, \
    compute_cost_test_case, backward_propagation_test_case, update_parameters_test_case, nn_model_test_case, \
    predict_test_case

X, Y = load_planar_dataset()
from NN_model import nn_model
from Predict import predict


# more model
# plt.figure(figsize=(16, 32))
hidden_layer_size = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_size):
    plt.subplot(5, 2, i + 1)
    plt.title("hidden layer of size %d " % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1-predictions.T)) / float(Y.size) * 100)
    print("Accuracy for {} hidden units: {}%".format(n_h, accuracy))
plt.show()
