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
    train_data, validation_data, test_data = mnist.load_datasets()
    print("Data loaded ...")

    # print(len(train_data))
    # print(len(validation_data))
    # print(len(test_data))
    # print(train_data[0].input())
    # print(train_data[0].target())

    input_size = 784
    hidden_layer_size = 50
    output_size = 10
    n_epochs = 2

    ann = ANN([input_size, hidden_layer_size, output_size])
    # print(ann.layers[0].output)
    # print(ann.layers[3].weights)

    ann.train(train_data, n_epochs)
    for i in range(len(validation_data)):
        print(ann.validate(validation_data[i].inputs))

    # print(validation_data[-1].targets)

    # for i in range(len(validation_data)):
    #     print(ann.test(validation_data[i].inputs))
    # ann.test(validation_data[i])


if __name__ == '__main__':
    main()
