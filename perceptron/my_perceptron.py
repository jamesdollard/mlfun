
import torch
import numpy as np


class MyPerceptron:

    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.w = 0
        self.b = 0

    def classify(self, x):
        if np.dot(x, self.w) + self.b > 0:
            return 1
        elif np.dot(x, self.w) + self.b < 0:
            return -1
        else:
            print("Right on hyperplane")
            return 0

    def fit(self, train, truth):

        # initialize weights and bias, lr copy
        self.w = torch.rand(len(train[0]))
        self.b = torch.rand(1)
        learn_rate = self.lr

        errors = True
        iterations = 0
        while errors and iterations < self.epochs:
            errors = False
            for i in range(len(train)):
                x = train[i]
                y = truth[i]

                # check if correctly classified
                if y * self.classify(x) < 0:
                    errors = True

                    # adjust weights
                    for j in range(len(self.w)):
                        self.w[j] += learn_rate * y * x[j]

                    # adjust bias
                        self.b += learn_rate * y
            #print("Weight: " + str(w))
            iterations += 1
            learn_rate = learn_rate * 0.95
        #print(iterations)
        return self.w, self.b

    def test(self, data, labels):
        correct = 0
        for i in range(len(data)):
            if self.classify(data[i]) == labels[i]:
                correct += 1

        return correct / len(data)
