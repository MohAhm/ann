import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mnist import Mnist
from ann import ANN


def main():
    """ EntryPoint """

    # Set Data ...
    df = pd.read_csv('assignment5.csv', delimiter=',')
    inputs = np.array(df.values.tolist())
    # print(inputs)

    s = pd.Series(list(range(10)))
    one_hot = pd.get_dummies(s)     # One-hot code
    targets = np.array(one_hot.values.tolist())
    # print(len(targets))

    mnist = Mnist(inputs, targets)
    training_set, validation_set, testing_set = mnist.load_datasets()

    # print(len(training_set))
    # print(len(validation_set))
    # print(len(testing_set))
    # print(training_set[0].input())
    # print(training_set[0].target())

    input_size = 784
    hidden_layer_size = 50
    output_size = 10
    n_epochs = 5

    ann = ANN([input_size, hidden_layer_size, hidden_layer_size, output_size])
    # print(ann.layers[0].output)
    # print(ann.layers[3].weights)

    for i in range(len(validation_set)):
        print(ann.test(validation_set[i]))
        # ann.test(validation_set[i])


if __name__ == '__main__':
    main()
