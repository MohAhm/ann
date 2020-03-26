
class Mnist:
    def __init__(self, examples):
        self.train = []
        self.test = []
        self.validation = []

        total_size = len(examples)
        set_count = 0
        # training set (70%)
        for i in range(int(round(total_size * 0.7))):
            self.train.append(examples[i][1:])
            set_count += 1

        # validation set (10%)
        for i in range(set_count, set_count + int(round(total_size * 0.1))):
            self.validation.append(examples[i][1:])
            set_count += 1

        # testing set (20%)
        for i in range(set_count, set_count + int(round(total_size * 0.2))):
            self.test.append(examples[i][1:])
