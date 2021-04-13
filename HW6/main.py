import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
from scipy.stats import multivariate_normal

# Size
N = 300
K = 5
# Means
class_means = np.array([[+2.5, +2.5],
                        [-2.5, +2.5],
                        [-2.5, -2.5],
                        [+2.5, -2.5],
                        [0, 0]])
# Covariance
class_covariances = np.array([[[0.8, -0.6], [-0.6, 0.8]],
                              [[0.8, 0.6], [0.6, 0.8]],
                              [[0.8, -0.6], [-0.6, 0.8]],
                              [[0.8, 0.6], [0.6, 0.8]],
                              [[1.6, 0], [0, 1.6]]])

# Randomly initialize points

np.random.seed(421)
X1 = np.random.multivariate_normal(class_means[0, :], class_covariances[0, :, :], 50)
X2 = np.random.multivariate_normal(class_means[1, :], class_covariances[1, :, :], 50)
X3 = np.random.multivariate_normal(class_means[2, :], class_covariances[2, :, :], 50)
X4 = np.random.multivariate_normal(class_means[3, :], class_covariances[3, :, :], 50)
X5 = np.random.multivariate_normal(class_means[4, :], class_covariances[4, :, :], 100)
Data = np.vstack((X1, X2, X3, X4, X5))

plt.plot(Data[:, 0], Data[:, 1], ".", markersize=10, color="black")
plt.show()


# Update Functions and Visualization


def update_centroids(memberships, X):
    if memberships is None:
        centroids = X[np.random.choice(range(N), K), :]
    else:
        centroids = np.vstack([np.mean(X[memberships == k,], axis=0) for k in range(K)])
    return centroids


def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis=0)
    return (memberships)


def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:, 0], X[:, 1], ".", markersize=10, color="black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
                     color=cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize=12,
                 markerfacecolor=cluster_colors[c], markeredgecolor="black")
    plt.xlabel("x1")
    plt.ylabel("x2")


# 2 iterations
centroids = None
memberships = None
iteration = 1
while True:
    print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    centroids = update_centroids(memberships, Data)
    if np.alltrue(centroids == old_centroids):
        break
    else:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plot_current_state(centroids, memberships, Data)

    old_memberships = memberships
    memberships = update_memberships(centroids, Data)
    if np.alltrue(memberships == old_memberships):
        plt.show()
        break
    else:
        plt.subplot(1, 2, 2)
        plot_current_state(centroids, memberships, Data)
        plt.show()
    if iteration == 2:
        break
    iteration = iteration + 1

covariances = []
for c in range(K):
    covariances.append(np.cov(np.transpose(Data[memberships == c])))

priors = []
for c in range(N):
    total = np.empty(0)
    temp_p = np.empty(0)
    for d in range(K):
        n = multivariate_normal(centroids[d], covariances[d]) \
                .pdf(Data[c]) * (Data[memberships == d].shape[0] / N)
        total = np.append(total, n)
    total_n = np.sum(total)
    for d in range(K):
        n = multivariate_normal(centroids[d], covariances[d]) \
                .pdf(Data[c]) * (Data[memberships == d].shape[0] / N)
        temp_p = np.append(temp_p, n / total_n)
    priors.append(temp_p)

cpr = []
for i in range(K):
    cpr.append(Data[memberships == i].shape[0] / N)


def em_e(Data, centroids, covariances, cpr):
    priors = []
    for c in range(N):
        total = np.empty(0)
        temp_p = np.empty(0)
        for d in range(K):
            n = multivariate_normal(centroids[d], covariances[d]) \
                    .pdf(Data[c]) * cpr[d]
            total = np.append(total, n)
        total_n = np.sum(total)
        for d in range(K):
            n = multivariate_normal(centroids[d], covariances[d]) \
                    .pdf(Data[c]) * cpr[d]
            temp_p = np.append(temp_p, n / total_n)
        priors.append(temp_p)
    return priors


def em_m(Data, centroids, covariances, priors):

    cpr = []
    for c in range(K):
        td = 0
        t = 0
        for d in range(N):
            td += priors[d][c] * Data[d]
            t += priors[d][c]
        centroids[c] = td/t

    for c in range(K):
        td = 0
        t = 0
        for d in range(N):
            td += priors[d][c]*(np.dot((Data[d].reshape(1,2) - centroids[c].reshape(1,2)).T,
                                       (Data[d].reshape(1,2) - centroids[c].reshape(1,2))))
            t += priors[d][c]
        covariances[c] = td/t
    for c in range(K):
        td = 0
        for d in range(N):
            td += priors[d][c]
        cpr.append(td/N)
    return centroids, covariances, cpr


for i in range(100):
    centroids, covariances, cpr = em_m(Data, centroids, covariances, priors)
    priors = em_e(Data, centroids, covariances, cpr)


print(centroids)
cm = np.empty(0)
for c in range(N):
    cm = np.append(cm, np.argmax(priors[c]))
    
plot_current_state(centroids, cm, Data)
plt.show()
