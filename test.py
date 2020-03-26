
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_file(filename):
    file_object = open(filename)
    data = []

    while True:
        line = file_object.readline()

        if not line:
            break

        elif line[0].isdigit():
            digit = line.split(',')
            label = int(digit[0])
            pixels = list(map(int, digit[1:]))
            data.append((label, pixels))

    file_object.close()
    return data


# data = read_file('assignment5.csv')
# print(data[0][1])

# mnist = []
# for _, digit in enumerate(data, 1):
#     mnist.append(Mnist(digit[0], digit[1]))

# print(mnist)

# data = pd.read_csv('assignment5.csv', delimiter=',', skiprows=[0])
df = pd.read_csv('assignment5.csv', delimiter=',')
# print(data)
# print(data[1])
data = df.values.tolist()
# print(data[0])
# print(len(data))

# mnist = []
# for _, digit in enumerate(data, 1):
#     mnist.append(Mnist(digit[0], digit[1:]))

# data = np.loadtxt('assignment5.csv', delimiter=',', skiprows=1)
# data = np.genfromtxt('assignment5.csv', delimiter=',', skip_header=1)
# print(data)

# img = data[40][1:]
# img = data[3000][1]
# print(img)
# print(len(img))

# image_reshape = np.reshape(img, (28, 28))
# print(image_reshape)
# plt.imshow(img, cmap="Greys")
# plt.show()


# data = list(range(1, 101))
# print(data)

x = int(round(len(data) * 0.7))
y = int(round(len(data) * 0.1))
z = int(round(len(data) * 0.2))
print(x)
print(y)
print(z)
# print(x+y+z)

# count = 0
# for i in range(int(round(len(data) * 0.7))):
#     data[i] = 1
#     count += 1
# # print(data)

# for i in range(count, count+int(round(len(data) * 0.1))):
#     data[i] = 0
#     count += 1
# # print(data)

# for i in range(count, count+int(round(len(data) * 0.2))):
#     data[i] = -1
# print(data)
