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
            # Initialize the weights using the He initialization
            self.weights[i] = np.random.randn(self.layers[i], self.layers[i - 1]) * (np.sqrt(2 / self.layers[i - 1]))
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
            self.outputs[i] = np.dot(self.weights[i], self.activations[i - 1]) + self.biases[i]
            self.activations[i] = self.relu(self.outputs[i])
        
        # Calculate the activation of the output layer using softmax
        self.outputs[l] = np.dot(self.weights[l], self.activations[l - 1]) + self.biases[l]
        self.activations[l] = self.softmax(self.outputs[l])

        return self.outputs[l]

    # def cross_entropy_loss(self, targets, softmax_activation, n):
    #     loss = -np.sum((targets * np.log(softmax_activation)),
    #                    axis=0, keepdims=True)
    #     cost = np.sum(loss, axis=1) / n

    #     return cost

    def backward_propagation(self, inputs, targets):
        l = self.n_layers - 1   # l corresponds to the last layer of the network
        n = inputs.shape[1]     # n corresponds to the size of the inputs set

        # Calculate partial derivatives of the error 
        # with respect to output layer 
        self.derivatives['dZ' + str(l)] = (self.activations[l] - targets) / n

        # Calculate partial derivatives of the error  
        # with respect to hidden layers
        for i in range(1, l):
            self.derivatives['dA' + str(l - i)] = np.dot(self.weights[l - i + 1].T, self.derivatives['dZ' + str(l - i + 1)])
            self.derivatives['dZ' + str(l - i)] = self.derivatives['dA' + str(l - i)] * self.relu_derivative(self.outputs[l - i])
        
        # Calculate partial derivatives of the weights and the biases
        self.derivatives['dW1'] = np.dot(self.derivatives['dZ1'], inputs.T)
        self.derivatives['db1'] = np.sum(self.derivatives['dZ1'], axis=1, keepdims=True)

        for i in range(2, self.n_layers):
            self.derivatives['dW' + str(i)] = np.dot(self.derivatives['dZ' + str(i)], self.activations[i - 1].T)
            self.derivatives['db' + str(i)] = np.sum(self.derivatives['dZ' + str(i)], axis=1, keepdims=True)

    def update_parameters(self):
        for i in range(1, self.n_layers):
            # Update weights
            self.weights[i] = self.weights[i] - (self.learning_rate * self.derivatives['dW' + str(i)])
            # Update biases
            self.biases[i] = self.biases[i] - (self.learning_rate * self.derivatives['db' + str(i)])

    def learning(self, train_inputs, train_targets, validation_inputs, validation_targets, n_epochs):   
        validation_accuracy = []
        epochs = []

        for epoch in range(n_epochs):
            self.forward_propagation(train_inputs)
            # self.cross_entropy_loss(train_targets, activation, n)

            self.backward_propagation(train_inputs, train_targets)
            self.update_parameters()

            if epoch % 10 == 0:
                accuracy = self.evaluate(validation_inputs, validation_targets)
                validation_accuracy.append(accuracy)
                epochs.append(epoch)

                print('Epoch: %i, Validation Accuracy: %f' % (epoch, accuracy))

        return validation_accuracy, epochs

    def evaluate(self, inputs, targets, mnist=None):
        output = self.forward_propagation(inputs)

        o_val = np.transpose(output)
        t_val = np.transpose(targets)

        total_accuracy = 0

        if mnist:
            occurrence = {}
            classe_accuracy = {}

            for i in range(t_val.shape[0]):
                n = mnist.get_label(t_val[i])

                occurrence[n] = occurrence.get(n, 0) + 1

                if np.argmax(t_val[i]) == np.argmax(o_val[i]):
                    total_accuracy += 1
                    classe_accuracy[n] = classe_accuracy.get(n, 0) + 1

            occurrence = collections.OrderedDict(sorted(occurrence.items()))
            classe_accuracy = collections.OrderedDict(
                sorted(classe_accuracy.items()))

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
