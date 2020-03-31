import numpy as np
import pandas as pd
from mnist import Mnist


df = pd.read_csv('assignment5.csv', delimiter=',')
examples = np.array(df.values.tolist())

mnist = Mnist(examples)

# print(mnist.train[0])
# print(len(mnist.train))

training_set, validation_set, testing_set = mnist.load_dataset()
# print(training_set)
# print(training_set[0].input())
print(len(training_set))
print(len(validation_set))
print(len(testing_set))
