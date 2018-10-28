import numpy as np
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset, plot_decision_boundary
import sklearn

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
