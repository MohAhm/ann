
class Digit:
    def __init__(self, targets, inputs):
        self.targets = targets
        self.inputs = inputs


class Mnist:
    def __init__(self, inputs, targets):
        self.train = []
        self.test = []
        self.validation = []

        one_hot = one_hot_encoding(targets)
        total_size = len(inputs)
        set_count = 0
        # training set (70%)
        for i in range(int(round(total_size * 0.7))):
            self.train.append(Digit(one_hot[inputs[i][0]], inputs[i][1:]))
            set_count += 1

        # validation set (10%)
        for i in range(set_count, set_count + int(round(total_size * 0.1))):
            self.validation.append(
                Digit(one_hot[inputs[i][0]], inputs[i][1:]))
            set_count += 1

        # testing set (20%)
        for i in range(set_count, set_count + int(round(total_size * 0.2))):
            self.test.append(Digit(one_hot[inputs[i][0]], inputs[i][1:]))

    def load_datasets(self):
        return self.train, self.validation, self.test


def one_hot_encoding(targets):
    one_hot = {}

    for i, binary in enumerate(targets):
        one_hot[i] = binary

    return one_hot
