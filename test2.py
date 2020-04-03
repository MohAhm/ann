import numpy as np
import pandas as pd
from mnist import Mnist


y = np.array([2.08037199e-10, 1.86160544e-07, 1.70746214e-04, 1.88076914e-08,
              3.11420748e-07, 9.96705283e-01, 3.11681868e-03, 9.66506607e-15,
              6.63406487e-06, 1.25963207e-09])

x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# test = cross_entropy_loss(x, y)
# print(test)

def cross_entropy_loss(x, y):
    return (-1.0 / len(x)) * sum(x * np.log(_y) + (1 - _x) * np.log(1 - _y) for _x, _y in zip(x, y))


loss = cross_entropy_loss([1, 0], [0.9, 0.3])
print(loss)


def softmax(self, x):
    return np.exp(x) / np.sum(np.exp(x))


def deriv_sigmoid(x):
    lgst = sigmoid(x)
    return (1 - lgst) * lgst


def cross_entropy_loss2(x, y):
    return -np.sum(x * np.log(y))


def cross_entropy(x):
    p = softmax(x)
    return


def cross_entropy(X, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector. 
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = softmax(X)

    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss
# df = pd.read_csv('assignment5.csv', delimiter=',')
# examples = np.array(df.values.tolist())

# mnist = Mnist(examples)

# # print(mnist.train[0])
# # print(len(mnist.train))

# training_set, validation_set, testing_set = mnist.load_dataset()
# # print(training_set)
# # print(training_set[0].input())
# print(len(training_set))
# print(len(validation_set))
# print(len(testing_set))
