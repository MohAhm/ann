import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# print(X)

my_lst = np.array(range(10))
print(my_lst)

init_lst = np.zeros(len(my_lst))
print(init_lst)

init_lst = my_lst
print(init_lst)


# weights1 = np.random.rand(X.shape[1], 40)
# print(weights1)

# weights2 = np.random.rand(4, 1)
# print(weights2)

# prev_layer_size = 50
# good_range = 1.0 / math.sqrt(prev_layer_size + 1)
# print(good_range)

# tragets = np.zeros(10)
# print(tragets)

# s = pd.Series(list(range(10)))
# tragets = pd.get_dummies(s)
# # print(tragets)

# examples = np.array(tragets.values.tolist())
# print(examples)
# # print(examples[5])


# def one_hot_encoding(examples):
#     targets = {}

#     for label, one_hot in enumerate(examples):
#         targets[label] = one_hot

#     return targets


# test = one_hot_encoding(examples)
# print(test[5])
# print(np.array(test))

# for i, one_hot in enumerate(examples):
#     print(i)
#     print(one_hot)

# hash = {}

# for i, one_hot in enumerate(len(examples)):
#     hash[i] = one_hot

# print(hash[0])

# def possible_moves(self):
#         return list(self.moves.keys())

#     def get_move(self, key):
#         return self.moves.pop(key)
