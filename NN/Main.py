import numpy as np
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset, plot_decision_boundary
import sklearn
from testCases import layer_sizes_test_case, initialize_parameters_test_case, forward_propagation_test_case, \
    compute_cost_test_case, backward_propagation_test_case, update_parameters_test_case, nn_model_test_case, \
    predict_test_case

#  The general methodology to build a Neural Network is to:
#     1. Define the neural network structure ( # of input units,  # of hidden units, etc).
#     2. Initialize the model's parameters
#     3. Loop:
#         - Implement forward propagation
#         - Compute loss
#         - Implement backward propagation to get the gradients
#         - Update parameters (gradient descent)


# import sklearn.linear_model

np.random.seed(1)  # set a seed so that the results are consistent

X, Y = load_planar_dataset()

plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
# plt.show()


shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]
print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' % (m))
print()

#  Simple Logistic Regression
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)
plot_decision_boundary(lambda x: clf.predict(x), X, np.squeeze(Y))
plt.title("Logistic regression")
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' % float(
    (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
      '% ' + "(percentage of correctly labelled datapoints)")
print()

# define model structure
from Layer_Sizes import layer_sizes

X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))
print()

# Initialize_parameters
from Initialize_Parameters import initialize_parameters

n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# the loop
from Forward_Propagation import forward_propagation

X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)
print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["A2"]), np.mean(cache["A2"]))
print()

# compute cost
from Compute_Cost import compute_Cost

A2, Y_assess, parameters = compute_cost_test_case()
print("cost = " + str(compute_Cost(A2, Y_assess, parameters)))
print()

# compute gradient
from Backward_Propagation import backward_propagation

parameters, cache, X1, Y1 = backward_propagation_test_case()
grads = backward_propagation(parameters, cache, X1, Y1)
print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))

# update grads
from Update_Parameters import update_parameters

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# nn_model
from NN_model import nn_model

X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# predict
from Predict import predict

parameters, X_assess = predict_test_case()
prediction = predict(parameters, X_assess)
print("prediction mean = " + str(np.mean(prediction)))
print()

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
plt.title("Decision Boundary for hideen layer size " + str(4))
# plt.show()

# accuracy
prediction = predict(parameters, X)
print(
    "Accuracy : %d" % float((np.dot(Y, prediction.T) + np.dot(1 - Y, (1 - prediction).T)) / float(Y.size) * 100) + "%")
