import numpy as np
import loglinear as ll

def classifier_output(x, params):
    # YOUR CODE HERE.
    for i in range(len(params) - 1):
        W, b = params[i]
        x = np.tanh(np.dot(x, W) + b)

    return ll.classifier_output(x, params[-1])

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    # YOU CODE HERE
    U, b_tag = params[-1]

    y_hot = np.zeros(U.shape[1])
    y_hot[y] = 1

    y_tag = classifier_output(x, params)
    loss = -np.log(y_tag[y])

    gradients = []

    list_z = [x]
    list_a = [x]

    for W, b in params:
        list_z.append(list_z[-1].dot(W) + b)
        list_a.append(np.tanh(list_z[-1]))

    diff = y_tag - y_hot
    gradients.insert(0, [np.array([list_a[-2]]).transpose().dot(np.array([diff])), diff])

    for i in (range(len(params) - 1))[::-1]:
        gb = gradients[0][1].dot(params[i + 1][0].transpose()) * (1 - np.power(list_a[i+1], 2))
        gW = np.array([list_z[i]]).transpose().dot(np.array([gb]))
        gradients.insert(0, [gW, gb])

    return loss, gradients

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.

    Assume a tanh activation function between all the layers.

    return:
    a list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []

    for i in range(len(dims) - 1):
        W = np.random.randn(dims[i], dims[i + 1]) / np.sqrt(dims[i])
        b = np.zeros(dims[i + 1])
        params.append([W, b])
    return params
