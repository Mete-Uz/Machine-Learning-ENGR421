import numpy as np
import math
import matplotlib.pyplot as plt

old_settings = np.seterr(all='ignore')

# read data into memory
data_set = np.genfromtxt("hw04_data_set.csv", delimiter=",", skip_header=1)

# initialize training and test sets
train_set = np.asarray(data_set[0:100, :])
test_set = np.asarray(data_set[100:133, :])
x_train = np.asarray(train_set[:, 0])
y_train = np.asarray(train_set[:, 1])
x_test = np.asarray(test_set[:, 0])
y_test = np.asarray(test_set[:, 1])

# Drawing Parameters
minimum_value = 0
maximum_value = 60
data_interval = np.linspace(minimum_value, maximum_value, 6001)

# Regressogram
bin_width = 3
left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)
p_hat = np.asarray([np.sum(((left_borders[b] < x_train) & (x_train <= right_borders[b])) * y_train) /
                    np.sum(((left_borders[b] < x_train) & (x_train <= right_borders[b])))
                    for b in range(len(left_borders))])

plt.figure(figsize=(10, 6))
plt.plot(train_set[:, 0], train_set[:, 1], "r.", markersize=10, label="training data")
plt.plot(test_set[:, 0], test_set[:, 1], "g.", markersize=10, label="test data")
plt.legend(loc="upper left")
plt.title('h = 3')
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")
plt.show()
N = test_set.shape[0]


def RMSE(test_set, y_test, p_hat, x_test, bin_width):
    RMSE = None
    N = test_set.shape[0]
    for c in range(N):
        temp = (y_test[c] - p_hat[(x_test[c] / bin_width).astype(int)]) ** 2
        if RMSE is None:
            RMSE = temp
        else:
            RMSE = np.append(RMSE, temp)

    RMSE = np.sqrt(np.sum(RMSE) / N)
    return RMSE


def RMSE_interval(test_set, y_test, p_hat, x_test):
    RMSE = None
    N = test_set.shape[0]
    for c in range(N):
        temp = (y_test[c] - p_hat[(x_test[c] * 100).astype(int)]) ** 2
        if RMSE is None:
            RMSE = temp
        else:
            RMSE = np.append(RMSE, temp)

    RMSE = np.sqrt(np.sum(RMSE) / N)
    return RMSE


RMSE1 = RMSE(test_set, y_test, p_hat, x_test, bin_width)
print("Regressogram: RMSE is", "{:.4f}".format(RMSE1), "when h is", bin_width)


# Running Mean Smoother

def weight(x):
    N = x.shape[0]
    for c in range(N):
        if np.abs(x[c]) < 1:
            x[c] = 1
        else:
            x[c] = 0
    return x


bin_width = 3
p_hat_RMS = np.asarray([np.sum(weight(((x - x_train) / (bin_width / 2))) * y_train) /
                        np.sum(weight(((x - x_train) / (bin_width / 2)))) for x in data_interval])

plt.figure(figsize=(10, 6))
plt.plot(train_set[:, 0], train_set[:, 1], "r.", markersize=10, label="training data")
plt.plot(test_set[:, 0], test_set[:, 1], "g.", markersize=10, label="test data")
plt.legend(loc="upper left")
plt.title('h = 3')
plt.plot(data_interval, p_hat_RMS, "k-")
plt.show()

RMSE2 = RMSE_interval(test_set, y_test, p_hat_RMS, x_test)
print("Running Mean Smoother: RMSE is", "{:.4f}".format(RMSE2), "when h is", bin_width)

# Kernel Smoothing
bin_width = 1
p_hat_K = np.asarray([np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train) ** 2 / bin_width ** 2) * y_train) /
                      np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train) ** 2 / bin_width ** 2))
                      for x in data_interval])

plt.figure(figsize=(10, 6))
plt.plot(train_set[:, 0], train_set[:, 1], "r.", markersize=10, label="training data")
plt.plot(test_set[:, 0], test_set[:, 1], "g.", markersize=10, label="test data")
plt.legend(loc="upper left")
plt.title('h = 1')
plt.plot(data_interval, p_hat_K, "k-")
plt.show()

RMSE3 = RMSE_interval(test_set, y_test, p_hat_K, x_test)
print("Kernel Smoother: RMSE is", "{:.4f}".format(RMSE3), "when h is", bin_width)
