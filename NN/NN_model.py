# -*- coding: utf-8 -*-
# @Time     : 2018/10/28 20:30
# @Author   : HuangYin
# @FileName : NN_model.py
# @Software : PyCharm
from Layer_Sizes import layer_sizes
from Initialize_Parameters import initialize_parameters
from Forward_Propagation import forward_propagation
from Compute_Cost import compute_Cost
from Backward_Propagation import backward_propagation
from Update_Parameters import update_parameters


def nn_model(X, Y, n_h, num_iterations=1000, print_cost=False):
    """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(num_iterations):
        # forward_propagation
        A2, cache = forward_propagation(X, parameters)
        # forward_propagation
        cost = compute_Cost(A2, Y, parameters)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i %f" % (i, cost))
        #     backward_propagation
        grads = backward_propagation(parameters, cache, X, Y)
        # update_parameters
        parameters = update_parameters(parameters, grads)
    return parameters
