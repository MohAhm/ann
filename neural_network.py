import numpy as np
from mnist import Mnist
import collections


class ANN:
    def __init__(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        self.learning_rate = 0.1

        # Parameters
        self.weights = {}       # W: weight matrix associated with the layers
        self.biases = {}        # b: vector of biases associated with the layers
        self.outputs = {}       # Z: vector of weighted outputs of the neurons
        self.activations = {}   # A: vector of activations of the neurons

        self.derivatives = {}   # d: derivatives of the parameters

        for i in range(1, self.n_layers):
            # Initialize the weights randomly using the He initialization
            self.weights[i] = np.random.randn(
                self.layers[i], self.layers[i - 1]) * (np.sqrt(2 / self.layers[i - 1]))
            # Initialize the biases to zeros
            self.biases[i] = np.zeros((self.layers[i], 1))

    def relu(self, x):
        # Activation function for the hidden layers
        return np.maximum(0, x)

    def relu_derivative(self, x):
        # Derivative of the Relu activation function
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def softmax(self, x):
        # Activation function for the output layer
        return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

    def forward_propagation(self, inputs):
        l = self.n_layers - 1   # l corresponds to the last layer of the network

        # Calculate the activation of hidden layers using Relu,
        # starting from the first layer using the inputs
        self.outputs[1] = np.dot(self.weights[1], inputs) + self.biases[1]
        self.activations[1] = self.relu(self.outputs[1])

        for i in range(2, l):
            self.outputs[i] = np.dot(
                self.weights[i], self.activations[i - 1]) + self.biases[i]
            self.activations[i] = self.relu(self.outputs[i])

        # Calculate the activation of the output layer using softmax
        self.outputs[l] = np.dot(
            self.weights[l], self.activations[l - 1]) + self.biases[l]
        self.activations[l] = self.softmax(self.outputs[l])

        return self.outputs[l]

    def backward_propagation(self, inputs, targets):
        l = self.n_layers - 1   # l corresponds to the last layer of the network
        n = inputs.shape[1]     # n the size of the inputs set

        # Calculate the gradients of the outputs starting from the last layer
        self.derivatives['dZ' + str(l)] = (self.activations[l] - targets) / n

        for i in range(1, l):
            self.derivatives['dA' + str(l - i)] = np.dot(
                self.weights[l - i + 1].T, self.derivatives['dZ' + str(l - i + 1)])
            self.derivatives['dZ' + str(l - i)] = self.derivatives['dA' + str(
                l - i)] * self.relu_derivative(self.outputs[l - i])

        # Calculate the gradients of the weights and biases
        self.derivatives['dW1'] = np.dot(self.derivatives['dZ1'], inputs.T)
        self.derivatives['dB1'] = np.sum(
            self.derivatives['dZ1'], axis=1, keepdims=True)

        for i in range(2, self.n_layers):
            self.derivatives['dW' + str(i)] = np.dot(
                self.derivatives['dZ' + str(i)], self.activations[i - 1].T)
            self.derivatives['dB' + str(i)] = np.sum(
                self.derivatives['dZ' + str(i)], axis=1, keepdims=True)

    def update_parameters(self):
        for i in range(1, self.n_layers):
            # Update weights
            self.weights[i] = self.weights[i] - \
                (self.learning_rate * self.derivatives['dW' + str(i)])
            # Update biases
            self.biases[i] = self.biases[i] - \
                (self.learning_rate * self.derivatives['dB' + str(i)])

    def learning(self, train_inputs, train_targets, validation_inputs, validation_targets, n_epochs):
        validation_accuracy = []
        epochs = []

        for epoch in range(n_epochs+1):
            # Training
            self.forward_propagation(train_inputs)
            self.backward_propagation(train_inputs, train_targets)
            self.update_parameters()

            # Validate the network every 10 epoch
            if epoch % 10 == 0:
                accuracy = self.evaluate(validation_inputs, validation_targets)
                validation_accuracy.append(accuracy)
                epochs.append(epoch)

                print('Epoch: %i, Validation accuracy: %f' % (epoch, accuracy))

        return validation_accuracy, epochs

    def evaluate(self, inputs, targets, mnist=None):
        outputs = self.forward_propagation(inputs)

        t_val = np.transpose(targets)
        o_val = np.transpose(outputs)

        total_accuracy = 0

        if mnist:
            # Store the occurrence of each class
            occurrence = {}
            classe_accuracy = {}

            for i in range(t_val.shape[0]):
                y = mnist.get_label(t_val[i])
                y_pred = mnist.get_pred_label(o_val[i])
                # print(o_val[i])

                occurrence[y] = occurrence.get(y, 0) + 1

                if y == y_pred:
                    total_accuracy += 1
                    classe_accuracy[y] = classe_accuracy.get(y, 0) + 1
                # else:
                #     print('Right: %i, Pred: %i' % (y, y_pred))

            occurrence = collections.OrderedDict(sorted(occurrence.items()))
            classe_accuracy = collections.OrderedDict(
                sorted(classe_accuracy.items()))

            # Calculate the accuracy for each class
            classes_accuracy = []
            for i in list(occurrence.keys()):
                classes_accuracy.append(
                    (classe_accuracy[i]/occurrence[i]) * 100)

            return (total_accuracy/t_val.shape[0]) * 100, classes_accuracy

        else:
            for i in range(t_val.shape[0]):
                if np.argmax(t_val[i]) == np.argmax(o_val[i]):
                    total_accuracy += 1

            return (total_accuracy/t_val.shape[0]) * 100
