
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from perceptron import my_perceptron
import random

# CONSTANTS
LR = 0.1
EPOCHS = 200

# Initialize perceptrons
my_p = my_perceptron.MyPerceptron(LR, EPOCHS)
sk_p = Perceptron(max_iter=EPOCHS, tol=1e-3)


# MY DATA

train = [[5, 5, 1], [6, 5, 0], [5, 5, 3], [1, 2, 1], [2, 2, 3], [0, 1, 2]]

labels = [1, 1, 1, -1, -1, -1]

my_p.fit(train, labels)
sk_p.fit(train, labels)
my_score = my_p.test(train, labels)
sk_score = sk_p.score(train, labels)

print("\nMy dataset:")
print("My perceptron score: " + str(my_score))
print("Sklearn perceptron score: " + str(sk_score))


# IRIS DATA STUFF

# Download iris dataset, define labels
iris = load_iris()
data = iris['data']

SETOSA = 0
VERSICOLOR = 1
VIRGINICA = 2


def make_iris_labels(flower):
    targets = iris['target']
    ret = []
    for target in targets:
        if target == flower:
            ret.append(1)
        else:
            ret.append(-1)
    return ret


# test setosa classifications, the only linearly separable classification
labels = make_iris_labels(SETOSA)

my_p.fit(data, labels)
sk_p.fit(data, labels)
my_score = my_p.test(data, labels)
sk_score = sk_p.score(data, labels)

print("\nSetosa dataset (linearly separable):")
print("My perceptron score: " + str(my_score))
print("Sklearn perceptron score: " + str(sk_score))

# test versicolor classifications (not linearly separable)
labels = make_iris_labels(VERSICOLOR)

my_p.fit(data, labels)
sk_p.fit(data, labels)
my_score = my_p.test(data, labels)
sk_score = sk_p.score(data, labels)

print("\nVersicolor dataset (non-linear):")
print("My perceptron score: " + str(my_score))
print("Sklearn perceptron score: " + str(sk_score))

# test virginica classification (also not linearly separable)
labels = make_iris_labels(VIRGINICA)

my_p.fit(data, labels)
sk_p.fit(data, labels)
my_score = my_p.test(data, labels)
sk_score = sk_p.score(data, labels)

print("\nVirginica dataset (non-linear):")
print("My perceptron score: " + str(my_score))
print("Sklearn perceptron score: " + str(sk_score))


# UCI digits 8x8 classification

# change epoch, reinitialize perceptrons
EPOCHS = 25
my_p = my_perceptron.MyPerceptron(LR, EPOCHS)
sk_p = Perceptron(max_iter=EPOCHS, tol=1e-3)

# download dataset, make labels
digits = load_digits()
data = digits['data']

ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6
SEVEN = 7
EIGHT = 8
NINE = 9
ZERO = 0

numbers = [ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, ZERO]

# train with 80% of data, test with 20%
x_y = []
for i in range(len(digits['data'])):
    x_y.append([digits['data'][i], digits['target'][i]])

random.shuffle(x_y)
four_fifths_idx = int(len(x_y) * 0.8)

train = x_y[:four_fifths_idx]
train_data = []
train_labels = []
for d in train:
    train_data.append(d[0])
    train_labels.append(d[1])

test = x_y[four_fifths_idx:]
test_data = []
test_labels = []
for d in test:
    test_data.append(d[0])
    test_labels.append(d[1])


def make_digit_data(number):
    ret_train = []
    ret_test = []
    for d in train_labels:
        if d == number:
            ret_train.append(1)
        else:
            ret_train.append(-1)
    for d in test_labels:
        if d == number:
            ret_test.append(1)
        else:
            ret_test.append(-1)

    return ret_train, ret_test


def train_test_digits():
    for number in numbers:
        train_labels_n, test_labels_n = make_digit_data(number)

        my_p.fit(train_data, train_labels_n)
        sk_p.fit(train_data, train_labels_n)
        my_score = my_p.test(test_data, test_labels_n)
        sk_score = sk_p.score(test_data, test_labels_n)
        print("\n" + str(number) + " dataset:")
        print("My perceptron score: " + str(my_score))
        print("Sklearn perceptron score: " + str(sk_score))


train_test_digits()
