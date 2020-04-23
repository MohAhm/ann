import numpy as np
import pandas as pd
from mnist import Mnist
import matplotlib.pyplot as plt


epoch = 10

for i in range(epoch):
    print(i)



# val = np.transpose(test_inputs)
# # print(val[0])

# fig = plt.figure(figsize=(6, 6))
# fig.subplots_adjust(left=0, right=1, bottom=0,
#                     top=1, hspace=0.05, wspace=0.05)

# n = np.min([25, len(val)])
# for i in range(n):
#     img = val[i]
#     img = img.reshape((28, 28))
#     fig.add_subplot(5, 5, i + 1)
#     # plt.title(i)
#     plt.text(0, 7, str(i), color='b')
#     plt.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
# plt.show()
