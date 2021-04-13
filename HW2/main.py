import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# safe logarithm
def safelog(x):
    return np.log(x + 1e-100)


# read data into memory
data_set = np.genfromtxt("hw02_data_set_images.csv", delimiter=",")
labels = pd.read_csv("hw02_data_set_labels.csv", header=None, names=["letter"])
label_set = pd.get_dummies(labels)
label_set = label_set.to_numpy()

# initialize training and test sets
train_set = np.empty([125, 320])
test_set = np.empty([70, 320])
train_labels = np.empty([125, 5])
test_labels = np.empty([70, 5])
trp = 0
tep = 0

# divide data in to their respective sets
N = len(data_set)
for c in range(0, N, 39):
    train_set[trp + 0:trp + 25] = data_set[c + 0:c + 25]
    trp += 25
    test_set[tep + 0:tep + 14] = data_set[c + 25:c + 39]
    tep += 14

trp = 0
tep = 0
for c in range(0, N, 39):
    train_labels[trp + 0:trp + 25] = label_set[c + 0:c + 25]
    trp += 25
    test_labels[tep + 0:tep + 14] = label_set[c + 25:c + 39]
    tep += 14


# declare function that will be used
def sigmoid(data, W, w0):
    return 1 / (1 + np.exp(-(np.matmul(data, W) + w0)))


def gradient_W(data, train_labels, label_predicted):
    W = np.asarray([-np.sum(np.repeat((train_labels[:,c] - label_predicted[:,c])[:, None], data.shape[1], axis = 1) *
                            data, axis = 0) for c in range(K)]).transpose()
    return W


def gradient_w0(train_labels, label_predicted):
    w0 = -np.sum(train_labels - label_predicted)
    return w0


K = 5

# learning parameters
eta = 0.01
epsilon = 1e-3

# randomly initialize W and w0
np.random.seed(60353)
W = np.random.uniform(low=-0.01, high=0.01, size=(train_set.shape[1], K))
w0 = np.random.uniform(low=-0.01, high=0.01, size=(1, K))

# iterative algorithm
iteration = 1
objective_values = []
while 1:
    label_predicted = sigmoid(train_set, W, w0)

    objective_values = np.append(objective_values, (np.sum(-(train_labels * safelog(label_predicted)) -
                                                           (1 - train_labels) * safelog(1 - label_predicted))))

    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(train_set, train_labels, label_predicted)
    w0 = w0 - eta * gradient_w0(train_labels, label_predicted)
    if np.sqrt(np.sum((w0 - w0_old)) ** 2 + np.sum((W - W_old) ** 2)) < epsilon:
        break

    iteration = iteration + 1

# plot iteration vs error
plt.figure(figsize=(10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

# confusion matrix for the training set
y_predicted = np.argmax(label_predicted, axis=1) + 1
y_truth = np.argmax(train_labels, axis=1) + 1
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames=['y_predicted'], colnames=['y_train'])
print(confusion_matrix)

# confusion matrix for the test set
test_predicted = sigmoid(test_set, W, w0)
ytest_predicted = np.argmax(test_predicted, axis=1) + 1
ytest_truth = np.argmax(test_labels, axis=1) + 1
confusion_matrix_test = pd.crosstab(ytest_predicted, ytest_truth, rownames=['y_predicted'], colnames=['y_test'])
print(confusion_matrix_test)
