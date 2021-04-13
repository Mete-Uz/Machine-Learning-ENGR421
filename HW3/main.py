import numpy as np
import pandas as pd
from PIL import Image as img
from PIL import ImageOps as iop


# safe logarithm
def safelog(x):
    return np.log(x + 1e-100)


# read data into memory
data_set = np.genfromtxt("hw03_data_set_images.csv", delimiter=",")
labels = pd.read_csv("hw03_data_set_labels.csv", header=None, names=["letter"])
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

y_truth = np.argmax(train_labels, axis=1) + 1

N = len(train_set) / 5
K = 5
# Get pcd
pcd = np.empty([320, 5])
for c in range(K):
    for d in range(320):
        pcd[d, c] = np.mean(train_set[y_truth == c + 1, d])

pcd_img = None
# resize pcd so each letter is 20 x 16, multiply with 255 to get 8 bit pixel values
for c in range(K):
    pcd_imgt = np.array(pcd[:, c] * 255).astype(int)
    pcd_imgt.resize((16, 20))
    if pcd_img is None:
        pcd_img = pcd_imgt
    else:
        pcd_img = np.concatenate((pcd_img, pcd_imgt), axis=0)

# some image manuplations to get desired image
estimg = img.fromarray(pcd_img.transpose()).convert('L')
estimg = iop.invert(estimg)
estimg.show()

# bernoulli naive bayes classifier
def naive_bayes(pcd, data):
    N = data.shape[0]
    scores = np.empty((N, 5))
    for d in range(N):
        for c in range(K):
            scores[d, c] = np.sum(((data[d, :] * safelog(pcd[:, c])) +
                                   ((1 - data[d, :]) * safelog(1 - (pcd[:, c])))))
    return scores

# print confusion matrices for test and training sets
train_scores = naive_bayes(pcd, train_set)
y_predicted = np.argmax(train_scores, axis=1) + 1
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames=['y_predicted'], colnames=['y_train'])
print(confusion_matrix)

test_score = naive_bayes(pcd, test_set)
ytest_predicted = np.argmax(test_score, axis=1) + 1
ytest_truth = np.argmax(test_labels, axis=1) + 1
confusion_matrix_test = pd.crosstab(ytest_predicted, ytest_truth, rownames=['y_predicted'], colnames=['y_test'])
print(confusion_matrix_test)
