import numpy as np

from layer import Layer
from mnist import Mnist


class ANN:
    def __init__(self, layer_sizes):
        self.layers = []
        self.learning_rate = 0.01

        for i in range(len(layer_sizes)):
            layer_size = layer_sizes[i]
            prev_layer_size = 0 if i == 0 else layer_sizes[i - 1]

            layer = Layer(i, layer_size, prev_layer_size)
            self.layers.append(layer)

    def train(self):
        pass

    def validate(self):
        pass

    def test(self, input):
        self.set_input(input)
        self.forward_propagation()

        return self.get_output()

    def set_input(self, input):
        input_layer = self.layers[0]

        for i in range(0, input_layer.n_neurons):
            input_layer.output[i] = input.pixels[i]

    def get_output(self):
        output_layer = self.layers[-1]
        output = np.zeros(output_layer.n_neurons)

        for i in range(len(output)):
            output[i] = output_layer.output[i]

        return output

    def forward_propagation(self):
        # exclude the last layer
        for i in range(len(self.layers) - 1):
            src_layer = self.layers[i]
            dst_layer = self.layers[i + 1]

            dst_layer.output = self.sigmoid(
                np.dot(src_layer.output, dst_layer.weights))

    def back_propagation(self):
        pass

    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))
