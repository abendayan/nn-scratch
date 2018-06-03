import numpy as np
import loglinear as ll

def classifier_output(x, params):
    W, b = params[0]
    # in params[1]: U, b_tag
    return ll.classifier_output(np.tanh(np.dot(x, W) + b), params[1])

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    W, b = params[0]
    U, b_tag = params[1]

    y_hot = np.zeros(U.shape[1])
    y_hot[y] = 1
    x_np = np.array(x)
    y_tag = classifier_output(x, params)
    loss = -np.log(y_tag[y])
    z = x.dot(W) + b
    a = np.tanh(z)

    gb_tag = y_tag - y_hot
    gU = np.outer(np.array([a]), np.array([gb_tag]))

    gb = gb_tag.dot(U.transpose()) * (1 - np.power(a, 2))
    gW = np.outer(x_np, np.array([gb]))
    return loss, [[gW, gb], [gU, gb_tag]]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    W = np.random.randn(in_dim, hid_dim) / np.sqrt(in_dim)
    b = np.zeros(hid_dim)
    U = np.random.randn(hid_dim, out_dim) / np.sqrt(hid_dim)
    b_tag = np.zeros(out_dim)
    params [[W, b], [U, b_tag]]
    return params
