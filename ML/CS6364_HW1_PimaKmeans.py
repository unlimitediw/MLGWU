import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy

np.random.seed(1)

train = pd.read_csv('../ML/data/CS6364HW1PIMA/diabetes.csv')

Y_train = train["Outcome"].values
X_train = train.drop(labels=['Outcome'], axis=1).values.reshape(-1, 8)

maxVal = 0
for i in range(len(X_train)):
    a = 7
    if X_train[i][a] > maxVal:
        maxVal = X_train[i][a]

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_validation, Y_validation, test_size=0.5, random_state=1)

Scaler = [0] * 8
datasize = len(X_train)
for i in range(datasize):
    for j in range(8):
        Scaler[j] += X_train[i][j]
for j in range(8):
    Scaler[j] /= datasize

# 8 features, first not consider weight, directly use KNN with normalization
centroid_list = np.ndarray([8, 2])
centroid_list.fill(0)
outcome_count = [0] * 2
for val in Y_train:
    outcome_count[val] += 1

'''
I want to convert the continuous features space to discrete features space
I believe that PCA will be better for the classification, but let me do it by hand first
pregnancies: the range is 0 to 17, it is ok
!It can be believe that apart from pregnancies, the feature with 0 val should be nan and miss
Glucose: it should be divided by 20 so the range is 0 to 9(maxVal is 199)
Blood pressure: it should be divided by 13 so that the range is 0 to 9(maxVal is 122)
SkinThickness: it should be divided by 10 so that the range is 0 to 9(maxVal is 99)
Insulin: 85, 0-9, 846
BMI: 7, 0-9, 67.1
DPF: 2.5, 0-9, 2.42
Disbete 9, 0-9, 81
'''

featureEntropyGain = [0] * 8
# (Num, lengthForEach)
dividedLength = [(18, 1), (10, 20), (10, 13), (10, 10), (10, 85), (10, 7), (10, 0.25), (10, 9)]


def featureNormalize(X):
    means = np.mean(X, axis=0)
    X_norm = X - means
    stds = np.std(X_norm, axis=0)
    myX_norm = X_norm / stds
    return means, stds, myX_norm


def getUSV(X):
    # covariance matrix formular
    cov_matrix = X.T.dot(X) / X.shape[0]
    U, S, V = scipy.linalg.svd(cov_matrix, full_matrices=True, compute_uv=True)
    return U, S, V


# nice PCA 用新的单位向量去生成新的少量投影数据
def projectData(X, U, K):
    # project only top "K" eigenvectors
    Ureduced = U[:, :K]
    z = X.dot(Ureduced)
    return z


# Scaler redesign
# Calculate information gain and use scaler divide it
def entropy(Y):
    get, notGet = 0, 0
    for val in Y:
        if val == 1:
            get += 1
        else:
            notGet += 1
    total = get + notGet
    thisEntropy = -(get / total) * math.log2(get / total) - (notGet / total) * math.log2(notGet / total)
    return thisEntropy


def setScaler(X, Y, totalEntropy):
    probabilitySet = []
    res = []
    # 该特征总数对总体样本量比例
    globalCount = []
    X_length = len(X)
    for f_idx in range(8):
        num = int(dividedLength[f_idx][0])
        gap = dividedLength[f_idx][1]
        # 8种特征情情况下的0，1该特征数量对特征总数比例
        localCount = [[0, 0] for _ in range(num)]
        nanCount = 0
        subGlobalCount = []
        for i in range(X_length):
            if X[i][f_idx] == 0:
                if f_idx != 0 and f_idx != 4:
                    nanCount += 1
                    continue
            cur = int(X[i][f_idx] / gap)
            if Y[i] == 1:
                localCount[cur][1] += 1
            else:
                localCount[cur][0] += 1
        for i in range(num):
            subGlobalCount.append(sum(localCount[i]) / (X_length - nanCount))
            total = localCount[i][0] + localCount[i][1]
            if total != 0:
                localCount[i][0] /= total
                localCount[i][1] /= total
        globalCount.append(subGlobalCount)
        probabilitySet.append(localCount)
    for i in range(8):
        cur = 0
        featureSize = len(probabilitySet[i])
        for subI in range(featureSize):
            if probabilitySet[i][subI][0] != 0 and probabilitySet[i][subI][1] != 0:
                cur -= globalCount[i][subI] * (probabilitySet[i][subI][0] * math.log2(probabilitySet[i][subI][0]) +
                                               probabilitySet[i][subI][1] * math.log2(probabilitySet[i][subI][1]))
        res.append(totalEntropy - cur)
    return res


# Use entropy gain as weight improve accuracy from 60.1%~66.9% to 70.6%~74.2%
# 1 level decision tree
entropy_cur = entropy(Y_train)
weight = setScaler(X_train, Y_train, entropy_cur)
# [0.06636496716491114, 0.19813817038328785, 0.02852341098885891, 0.08132480477656578, 0.16436404992575915, 0.09088716458855839, 0.02962067045270078, 0.08381789015716223]
Scaler[1] /= 3


# If I use ID3 algorithm
# n features, m gaps. nmlogmn

# training
def train(X, Y, centroids, count):
    length_data = len(X)
    length_features = 8
    for i in range(length_data):
        for f_idx in range(length_features):
            # use Scaler to normalize
            centroids[f_idx][Y[i]] += X[i][f_idx]

    for f_idx in range(8):
        # use outcome count to get the mean val
        centroids[f_idx][0] /= count[0]
        centroids[f_idx][1] /= count[1]

    return centroids


def validation(X_val, Y_val, centroids):
    correct_predict = 0
    for i in range(len(X_val)):
        count_panel = [0] * 2
        for f_idx in range(8):
            count_panel[0] += ((X_val[i][f_idx] - centroids[f_idx][0]) / Scaler[f_idx]) ** 2
            count_panel[1] += ((X_val[i][f_idx] - centroids[f_idx][1]) / Scaler[f_idx]) ** 2
        predic_idx = 0 if count_panel[0] < count_panel[1] else 1
        if predic_idx == Y_val[i]:
            correct_predict += 1
    return correct_predict / len(X_val)


centroid_list = train(X_train, Y_train, centroid_list, outcome_count)
print(validation(X_test, Y_test, centroid_list))
