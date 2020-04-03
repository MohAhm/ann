import numpy as np


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(int)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def softmax_derivative(x, y):
    val = softmax(x)
    return val - y


class ANN:
    def __init__(self, input_shape, n_neurons, output_shape):
        self.input = np.zeros(input_shape)
        self.layer1 = np.zeros(n_neurons)
        self.y = np.zeros(output_shape)

        self.weights1 = np.random.rand(self.input.shape[1], n_neurons)
        self.weights2 = np.random.rand(n_neurons, self.y.shape[1])

        self.output = np.zeros(self.y.shape)

    def train(self, x, y, n_epochs):
        for _ in range(n_epochs):
            self.set_input(x)
            self.set_target(y)

            self.feedforward()
            self.backprop()

    def test(self, x, y):
        self.set_input(x)
        self.set_target(y)

        self.feedforward()

        return self.get_output()

    def set_input(self, x):
        self.input = x

    def set_target(self, y):
        self.y = y

    def get_output(self):
        return self.output

    def feedforward(self):
        self.layer1 = relu(np.dot(self.input, self.weights1))
        self.output = softmax(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find the derivation of the
        # loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output)
                                            * softmax_derivative(self.output, self.y)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * softmax_derivative(
            self.output, self.y), self.weights2.T) * relu_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    # X = np.array([[0, 0, 1],
    #               [0, 1, 1],
    #               [1, 0, 1],
    #               [1, 1, 1]])
    # y = np.array([[0], [1], [1], [0]])

    X = np.array([[0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                      1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                      1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0,
                   1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]])

    y = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    input_shape = X.shape
    n_neurons = 20
    output_shape = y.shape
    n_epochs = 10000

    nn = ANN(input_shape, n_neurons, output_shape)

    nn.train(X, y, n_epochs)

    print(nn.test(X, y))
    # for i in range(n_epochs):
    #     nn.feedforward()
    #     nn.backprop()

    # print(nn.output)
