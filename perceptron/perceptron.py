import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# Download iris dataset
iris = load_iris()

# Load iris data, labels
data = iris['data']
targets = iris['target']

SETOSA = 0
VERSICOLOR = 1
VIRGINICA = 2


def classify_data(flower):
    ret = []
    for target in targets:
        if target == flower:
            ret.append(1)
        else:
            ret.append(-1)
    return ret

# CONSTANTS
LR = 0.1
EPOCHS = 100


def h_classify(x, w, b):
    if np.dot(x, w) + b > 0:
        return 1
    elif np.dot(x, w) + b < 0:
        return -1
    else:
        print("Right on hyperplane")
        return 0


def fit(train, truth):

    # initialize weights and bias
    w = torch.rand(len(train[0]))
    b = torch.rand(1)
    lr = LR

    errors = True
    iterations = 0
    while errors and iterations < EPOCHS:
        errors = False
        for i in range(len(train)):
            x = train[i]
            y = truth[i]

            # check if correctly classified
            if y * h_classify(x, w, b) < 0:
                errors = True

                # adjust weights
                for j in range(len(w)):
                    w[j] += lr * y * x[j]

                # adjust bias
                    b += lr * y
        #print("Weight: " + str(w))
        iterations += 1
        lr = lr * 0.95
    #print(iterations)
    return w, b


# # MY DATA
#
# train = [torch.tensor([5, 5, 1]), torch.tensor([6, 5, 0]), torch.tensor([5, 5, 3]),
#          torch.tensor([1, 2, 1]), torch.tensor([2, 2, 3]), torch.tensor([0, 1, 2])]
# labels = [1, 1, 1, -1, -1, -1]
# b = -3
# W = torch.tensor([0.5, 0.5, 1])
#
# print(fit(train, labels))


# IRIS DATA STUFF

# test setosa classifications, the only linearly separable classification on my perceptron vs sklearn
labels = classify_data(SETOSA)
W, b = fit(data, labels)

correct = 0
for i in range(50):
    if h_classify(data[i], W, b) == 1:
        correct += 1
for i in range(50, 150):
    if h_classify(data[i], W, b) == -1:
        correct += 1

p = Perceptron()
p.fit(data, labels)

print("\nSetosa dataset (linearly separable):")
print("My perceptron score: " + str(correct / 150))
print("Sklearn perceptron score: " + str(p.score(data, labels)))

# test versicolor classifications (not linearly separable) on my perceptron vs sklearn
labels = classify_data(VERSICOLOR)
W, b = fit(data, labels)

correct = 0
for i in range(50):
    if h_classify(data[i], W, b) == -1:
        correct += 1
for i in range(50, 100):
    if h_classify(data[i], W, b) == 1:
        correct += 1
for i in range(100, 150):
    if h_classify(data[i], W, b) == -1:
        correct += 1

p = Perceptron()
p.fit(data, labels)

print("\nVersicolor dataset (non-linear):")
print("My perceptron score: " + str(correct / 150))
print("Sklearn perceptron score: " + str(p.score(data, labels)))
