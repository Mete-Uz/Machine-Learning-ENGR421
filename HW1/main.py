import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Parameters
np.random.seed(60353)
# Means
class_means = np.array([[+0.0, +2.5],
                        [-2.5, -2.0],
                        [+2.5, -2.0]])
# Covariance
class_covariances = np.array([[[+3.2, +0.0],
                               [+0.0, +1.2]],
                              [[+1.2, -0.8],
                               [-0.8, +1.2]],
                              [[+1.2, +0.8],
                               [+0.8, +1.2]]])
# Size
class_sizes = np.array([120, 90, 90])

# Data generation
points1 = np.random.multivariate_normal(class_means[0, :], class_covariances[0, :, :], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1, :], class_covariances[1, :, :], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2, :], class_covariances[2, :, :], class_sizes[2])
X = np.vstack((points1, points2, points3))
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))
K = np.max(y)
N = X.shape[0]
# Estimations

# Means
means = [np.mean(X[:, 0][y == (c + 1)]) for c in range(K)]
means2 = [np.mean(X[:, 1][y == (c + 1)]) for c in range(K)]
sample_means = np.vstack((means, means2))

# Covariances
cov1 = np.cov(points1, rowvar=False)
cov2 = np.cov(points2, rowvar=False)
cov3 = np.cov(points3, rowvar=False)
sample_covariances = np.array(([cov1], [cov2], [cov3]))

# Prior probabilities
class_priors = [np.mean(y == (c + 1)) for c in range(K)]


def get_W(cov):
    W = np.dot(-0.5,  np.linalg.inv(cov))
    return W


def get_w(cov, mean):
    w = np.matmul(np.linalg.inv(cov), mean)
    return w


def get_w0(cov, mean, prior):
    temp1 = np.dot(-0.5, np.matmul(np.matmul(np.transpose(mean), np.linalg.inv(cov)), mean))
    temp2 = np.dot(-0.5, np.log(np.linalg.det(cov)))
    temp3 = np.log(prior)
    w0 = temp1 + temp2 + temp3
    return w0


def quadratic_disc(point, W, w, w0):
    disc = np.matmul(np.matmul(np.transpose(point), W), point) + np.matmul(np.transpose(w), point) + w0
    return disc


W = get_W([sample_covariances[c, :, :] for c in range(K)])
w = None
w0 = None

for c in range(K):
    w_sample = get_w(sample_covariances[c, :, :], sample_means[:, c:(c + 1)])
    if w is None:
        w = w_sample
    else:
        w = np.append(w, w_sample, axis=0)

    w0_sample = get_w0(sample_covariances[c, :, :], sample_means[:, c:(c + 1)], class_priors[c])
    if w0 is None:
        w0 = w0_sample
    else:
        w0 = np.append(w0, w0_sample)


scores_1 = None
scores_2 = None
scores_3 = None


for c in range(N):
    score = quadratic_disc(X[c, :], W[0, :, :], w[0, :], w0[0])
    if scores_1 is None:
        scores_1 = score
    else:
        scores_1 = np.append(scores_1, score)

for c in range(N):
    score = quadratic_disc(X[c, :], W[1, :, :], w[1, :], w0[1])
    if scores_2 is None:
        scores_2 = score
    else:
        scores_2 = np.append(scores_2, score)

for c in range(N):
    score = quadratic_disc(X[c, :], W[2, :, :], w[2, :], w0[2])
    if scores_3 is None:
        scores_3 = score
    else:
        scores_3 = np.append(scores_3, score)

scores = np.array([scores_1, scores_2, scores_3])

y_predicted = np.zeros(N).astype(int)
for c in range(N):
    y_predicted[c] = np.argmax(scores[:, c]) + 1

Y_predicted = np.zeros((N, K)).astype(int)
Y_predicted[range(N), y_predicted - 1] = 1

confusion_matrix = pd.crosstab(y_predicted, y, rownames= ['y_pred'], colnames= ['y_truth'])
print(confusion_matrix)

data_interval1 = np.linspace(-5, +5, 301)
data_interval2 = np.linspace(-5, +5, 301)
grid = None

for t in range(301):
    for r in range(301):
        data_grid = np.array([data_interval1[t], data_interval2[r]])
        if grid is None:
            grid = data_grid
        else:
            grid = np.vstack([grid, data_grid])

L = 90601
bound_1 = None
bound_2 = None
bound_3 = None

for c in range(L):
    score = quadratic_disc(grid[c, :], W[0, :, :], w[0, :], w0[0])

    if bound_1 is None:
        bound_1 = score
    else:
        bound_1 = np.append(bound_1, score)

for c in range(L):
    score = quadratic_disc(grid[c, :], W[1, :, :], w[1, :], w0[1])
    if bound_2 is None:
        bound_2 = score
    else:
        bound_2 = np.append(bound_2, score)

for c in range(L):
    score = quadratic_disc(grid[c, :], W[2, :, :], w[2, :], w0[2])
    if bound_3 is None:
        bound_3 = score
    else:
        bound_3 = np.append(bound_3, score)

boundaries = np.array([bound_1, bound_2, bound_3])

determinant_values = np.zeros(L).astype(int)
b_predicted = np.zeros(L).astype(int)
for c in range(L):
    b_predicted[c] = np.argmax(boundaries[:, c]) + 1

for c in range(L-1):
    temp = b_predicted[c]
    if b_predicted[c+1] != b_predicted[c]:
        determinant_values[c] = 1


plt.figure(figsize=(10, 10))

plt.plot(points1[:, 0], points1[:, 1], "r.", markersize=10)
plt.plot(points2[:, 0], points2[:, 1], "g.", markersize=10)
plt.plot(points3[:, 0], points3[:, 1], "b.", markersize=10)
plt.plot(X[y_predicted != y, 0], X[y_predicted != y, 1], "ko", markersize = 12, fillstyle = "none")
plt.plot(grid[determinant_values == 1, 0], grid[determinant_values == 1, 1], "k.")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

