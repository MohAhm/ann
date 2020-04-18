import numpy as np
import pandas as pd
from mnist import Mnist

import collections


df = pd.read_csv('assignment5.csv', delimiter=',')
dataset = df.values.tolist()

s = pd.Series(list(range(10)))
dummy = pd.get_dummies(s)
binaries = dummy.values.tolist()


mnist = Mnist(dataset, binaries)
_, _, test_targets = mnist.load_targets()

t = np.transpose(test_targets)

# print(t.shape[0])
# print(t[5])
# print(mnist.get_label(t[5]))
# print(len(t))

# a = []
# for i in range(len(t)):
#     a.append(mnist.get_label(t[i]))

# print(len(a))

occurrence = {}
accuracy = {}

# for i, n in enumerate(a):
for i in range(t.shape[0]):
    occurrence[mnist.get_label(t[i])] = occurrence.get(
        mnist.get_label(t[i]), 0) + 1

    if True:
        accuracy[mnist.get_label(t[i])] = accuracy.get(
            mnist.get_label(t[i]), 0) + 1

# print(occurrence)
# print(accuracy)

occurrence = collections.OrderedDict(sorted(occurrence.items()))
accuracy = collections.OrderedDict(sorted(accuracy.items()))
print(occurrence)
# print(accuracy[2])

test = []
for i in list(occurrence.keys()):
    test.append((accuracy[i]/occurrence[i]) * 100)

print(test)
print(list(occurrence.keys()))

# occurrence = {}

# for n in range(t.shape[0]):
#     occurrence[mnist.get_label(t[n])] = occurrence.get(n, 0) + 1

# print(occurrence)
