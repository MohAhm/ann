import numpy as np
import math
import numpy as np


class Layer:

    def __init__(self, id, layer_size, prev_layer_size):
        self.n_neurons = layer_size
        self.input = np.zeros(self.n_neurons)
        self.output = np.zeros(self.n_neurons)

        init_range = 1.0 / math.sqrt(prev_layer_size + 1)
        self.weights = np.random.uniform(-init_range,
                                         init_range, size=(prev_layer_size, self.n_neurons))

    