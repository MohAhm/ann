import pandas as pd
from mnist import Mnist


data = pd.read_csv('assignment5.csv', delimiter=',')
examples = data.values.tolist()

mnist = Mnist(examples)
print(len(mnist.validation))
