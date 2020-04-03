import numpy as np

from layer import Layer
from mnist import Mnist


class ANN:
    def __init__(self, layer_sizes):
        self.layers = []
        self.learning_rate = 0.1

        for i in range(len(layer_sizes)):
            layer_size = layer_sizes[i]
            prev_layer_size = 0 if i == 0 else layer_sizes[i - 1]

            layer = Layer(i, layer_size, prev_layer_size)
            self.layers.append(layer)

    def train(self, dataset, n_epochs):
        for epoch in range(n_epochs):
            epoch_err = 0  # testing

            for i in range(len(dataset)):
                self.set_input(dataset[i].inputs)
                self.forward_propagation()

                # for testing
                sample_err = self.cost(dataset[i].targets)
                epoch_err += sample_err

                self.backward_propagation()
                self.update_weights()

            # testing
            # if epoch % 100 == 0:
            # print(epoch, epoch_err)

    def cost(self, target):
        sample_err = 0  # testing
        output_layer = self.layers[-1]

        # print(target)
        # print(output_layer.output)
        for i in range(output_layer.n_neurons):
            # neuron_output = output_layer.output[i]
            # neuron_err = target[i] - neuron_output
            # print(neuron_err)
            # print(self.sigmoid_derivative(output_layer.input[i]))
            output_layer.y[i] = self.softmax_derivative(
                output_layer.input[i], target[i])  # * neuron_err

            print(output_layer.y[i])

        #     sample_err += neuron_err * neuron_err  # testing
        # sample_err *= 0.5  # testing

        return sample_err

    def update_weights(self):
        for i in range(1, len(self.layers)):
            for j in range(self.layers[i].n_neurons):
                self.layers[i].weights = self.layers[i].weights + self.learning_rate * \
                    np.dot(self.layers[i].y, self.layers[i-1].output[j])

                # print(self.layers[i].weights)

    def validate(self, input):
        self.set_input(input)
        self.forward_propagation()

        return self.get_output()

    def test(self, input):
        self.set_input(input)
        self.forward_propagation()

        return self.get_output()

    def set_input(self, input):
        input_layer = self.layers[0]

        for i in range(input_layer.n_neurons):
            input_layer.output[i] = input[i]

    def get_output(self):
        output_layer = self.layers[-1]
        output = np.zeros(output_layer.n_neurons)

        for i in range(len(output)):
            output[i] = output_layer.output[i]

        return output

    def forward_propagation(self):
        for is_last_element, i in self.signal_last(range(len(self.layers) - 1)):
            src_layer = self.layers[i]
            dst_layer = self.layers[i + 1]
            in_val = np.dot(src_layer.output, dst_layer.weights)

            # dst_layer.input = in_val
            # dst_layer.output = self.sigmoid(in_val)

            if is_last_element:
                dst_layer.output = self.stable_softmax(in_val)
            else:
                dst_layer.output = self.relu(in_val)

    def backward_propagation(self):
        for is_first_element, i in self.signal_first(range(len(self.layers) - 1, 0, -1)):
            # for i in range(len(self.layers) - 1, 0, -1):
            src_layer = self.layers[i]
            dst_layer = self.layers[i - 1]
            in_val = np.dot(src_layer.weights, src_layer.y)

            # if is_first_element:
            #     dst_layer.y = in_val * \
            #         self.softmax_derivative(dst_layer.input, dst_layer.y)
            # else:
            dst_layer.y = in_val * self.relu_derivative(dst_layer.input)

    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        val = self.sigmoid(x)
        return (1 - val) * val

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(int)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def softmax_derivative(self, x, y):
        val = self.stable_softmax(x)
        return val - y

    def stable_softmax(self, x):
        val = np.exp(x - np.max(x))
        return val / np.sum(val)

    def cross_entropy_loss(self, x, y):
        return -np.sum(x * np.log(y))

    def signal_last(self, lst):
        iterable = iter(lst)
        ret_var = next(iterable)

        for val in iterable:
            yield False, ret_var
            ret_var = val

        yield True, ret_var

    def signal_first(self, it):
        iterable = iter(it)
        yield True, next(iterable)
        for val in iterable:
            yield False, val
