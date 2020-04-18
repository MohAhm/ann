import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mnist import Mnist
from neural_network import ANN


def main():
    """ EntryPoint """

    # Set Data ...
    df = pd.read_csv('assignment5.csv', delimiter=',')
    dataset = df.values.tolist()
    # print(dataset)

    s = pd.Series(list(range(10)))
    dummy = pd.get_dummies(s)
    one_hot = dummy.values.tolist()
    # print(one_hot[0])

    mnist = Mnist(dataset, one_hot)
    train_inputs, validation_inputs, test_inputs = mnist.load_inputs()
    train_targets, validation_targets, test_targets = mnist.load_targets()
    print("Data loaded ...")
    # print(train_inputs[0])
    
    input_size = train_inputs.shape[0] # 784
    output_size = train_targets.shape[0] #10
    hidden_layer_size = 50

    ann = ANN([input_size, hidden_layer_size, hidden_layer_size, output_size])

    n_epochs = 150

    validation_accuracy, epochs = ann.learning(
        train_inputs, train_targets, validation_inputs, validation_targets, n_epochs)
    
    test_accuracy, classes_accuracy = ann.evaluate(
        test_inputs, test_targets, mnist)

    # for i in range(10):
    #     print('Class %i: %f' % (i, classes_accuracy[i]))

    print('Test Accuracy:', test_accuracy)

    plt.figure(1)
    plt.plot(epochs, validation_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('Validation Accuracy')

    plt.figure(2) 
    y_pos = np.arange(len(range(10)))
    plt.bar(y_pos, classes_accuracy)
    plt.xticks(y_pos, range(10))
    plt.ylabel('Accuracy')
    plt.title('Accuracy for each Class')
    plt.show()


if __name__ == '__main__':
    main()
