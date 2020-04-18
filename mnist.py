import numpy as np


class Mnist:
    def __init__(self, dataset, one_hot):
        self.train_inputs = []
        self.validation_inputs = []
        self.test_inputs = []

        self.train_targets = []
        self.validation_targets = []
        self.test_targets = []

        self.one_hot = {}
        self.one_hot_encoding(one_hot)

        total_size = len(dataset)
        set_count = 0

        # Training set (70%)
        for i in range(int(round(total_size * 0.7))):
            self.train_inputs.append(dataset[i][1:])
            self.train_targets.append(self.one_hot[dataset[i][0]])
            set_count += 1

        # Validation set (10%)
        for i in range(set_count, set_count + int(round(total_size * 0.1))):
            self.validation_inputs.append(dataset[i][1:])
            self.validation_targets.append(self.one_hot[dataset[i][0]])
            set_count += 1

        # Testing set (20%)
        for i in range(set_count, set_count + int(round(total_size * 0.2))):
            self.test_inputs.append(dataset[i][1:])
            self.test_targets.append(self.one_hot[dataset[i][0]])

        self.train_inputs = np.array(self.train_inputs, dtype='f')
        self.validation_inputs = np.array(self.validation_inputs, dtype='f')
        self.test_inputs = np.array(self.test_inputs, dtype='f')

        self.train_targets = np.array(self.train_targets, dtype='f')
        self.validation_targets = np.array(self.validation_targets, dtype='f')
        self.test_targets = np.array(self.test_targets, dtype='f')

    def load_inputs(self):
        # Transpose the data, turn row vector into column vector, and also 
        # Scale the data to make the values numerically stable (between 0 and 1)
        return self.train_inputs.T / 255, self.validation_inputs.T / 255, self.test_inputs.T / 255

    def load_targets(self):
        # Transpose the data, turn row vector into column vector
        return self.train_targets.T, self.validation_targets.T, self.test_targets.T

    def one_hot_encoding(self, one_hot):
        # Map the labels to one hot representations 
        for i, n in enumerate(one_hot):
            self.one_hot[i] = n

    def get_label(self, binary):
        # Get the label of the one hot
        for label, value in self.one_hot.items():
            if np.array_equal(value, binary):
                return label
