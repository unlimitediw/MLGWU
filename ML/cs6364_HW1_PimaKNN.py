# Tool
import math
from heapq import *

# ML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold

np.random.seed(1)


'''
Part of weight adjust
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
AGE: 9, 0-9, 81
'''

# (Num, lengthForEach)
dividedLength = [(18, 1), (10, 20), (10, 13), (10, 10), (10, 85), (10, 7), (10, 0.25), (10, 9)]


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


'''
Construction part*********************************************************************************************************************************************
'''

# fill nan data with median, not for 0 and 4, pregnancy and insulin
# after applying this part, the precision is down for 2~3%, so I have reason to believe that the person who provide
# nan data may not in a standard situation in that features.
'''
for i in range(len(X_train)):
    for f_idx in range(8):
        if f_idx != 0 and f_idx != 4 and X_train[i][f_idx] == 0:
            X_train[i][f_idx] = feature_median[f_idx]

for i in range(len(X_validation)):
    for f_idx in range(8):
        if f_idx != 0 and f_idx != 4 and X_validation[i][f_idx] == 0:
            X_validation[i][f_idx] = feature_median[f_idx]
'''


# KNN初始测试
# 不需要训练
def validation(X, Y, X_val, Y_val, k_val):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    correct = 0
    val_total = len(X_val)
    train_total = len(X)
    for v_idx in range(val_total):
        heap = []
        count = 0
        balance = 0
        predict_Y = 0
        for t_idx in range(train_total):
            curDistance = 0
            for f_idx in range(8):
                curDistance -= ((X[t_idx][f_idx] - X_val[v_idx][f_idx]) * weight[f_idx] / Scaler[f_idx]) ** 2
                # print(((X[t_idx][f_idx] - X_val[v_idx][f_idx]))**2)
            if count == k_val:
                if curDistance > heap[0][0]:
                    balance -= heappop(heap)[1]
                    balance += Y[t_idx]
                    heappush(heap, (curDistance, Y[t_idx]))
            else:
                count += 1
                balance += Y[t_idx]
                heappush(heap, (curDistance, Y[t_idx]))
            # print("***************")
        if balance > k_val // 2:
            predict_Y = 1
        if predict_Y == Y_val[v_idx]:
            correct += 1
    return correct / val_total


train = pd.read_csv('../ML/data/CS6364HW1PIMA/diabetes.csv')

#for k in range(1,20):
#    print(validation(X_train, Y_train, X_test, Y_test, k))

# loop k test
for k in range(21,22):
    totalTest, totalVal = 0, 0
    seed = 0
    Y_train = train["Outcome"].values
    X_train = train.drop(labels=['Outcome'], axis=1).values.reshape(-1, 8)

    k_fold = KFold(n_splits=10)
    TP, TN, FP, FN = 0,0,0,0
    for train_index,test_index in k_fold.split(X_train):
        Y_train = train["Outcome"].values
        X_train = train.drop(labels=['Outcome'], axis=1).values.reshape(-1, 8)
        X_train, X_test = X_train[train_index],X_train[test_index]
        Y_train, Y_test = Y_train[train_index],Y_train[test_index]
        Q = X_train.T
        Q_test = X_test.T
        Scaler = [1] * 8
        for i in range(8):
            zeroCount = 0
            totalCount = len(Q[i])
            for val in Q[i]:
                if val == 0 and (i == 2 or i == 3 or i == 4):
                    zeroCount += 1
            if 1 == 0:
                mean = np.mean(Q[i])
                std = np.std(Q[i])
                Q[i] = (Q[i] - mean) / std
                mean = np.mean(Q_test[i])
                std = np.std(Q_test[i])
                Q_test[i] = (Q_test[i] - mean) / std
            if 1 == 1:
                Scaler[i] = np.median(Q[i]) * (totalCount - zeroCount) / totalCount
            if 1 == 0:
                min = np.min(Q[i])
                max = np.max(Q[i])
                Q[i] = (Q[i] - min) / (max - min)
                min = np.min(Q_test[i])
                max = np.max(Q_test[i])
                Q_test[i] = (Q_test[i] - min) / (max - min)
            if i == 2 or i == 3 or i == 4:
                for j in range(len(Q[i])):
                    if Q[i][j] == 0:
                        Q[i][j] = Scaler[i]
                        pass
                for j in range(len(Q_test[i])):
                    if Q_test[i][j] == 0:
                        Q_test[i][j] = Scaler[i]
                        pass

        # 8 features, first not consider weight, directly use KNN with normalization
        outcome_count = [0] * 2
        feature_median = [0] * 8
        X_train_tran = X_train.T
        for i in range(len(X_train)):
            outcome_count[Y_train[i]] += 1
        for f_idx in range(8):
            feature_median[f_idx] = np.median(X_train_tran[f_idx])

        # Use entropy gain as weight improve accuracy from 60.1%~66.9% to 70.6%~74.2%
        # 1 level decision tree
        entropy_cur = entropy(Y_train)
        weight = setScaler(X_train, Y_train, entropy_cur)
        a = validation(X_train, Y_train, X_test, Y_test, k)
        totalTest += a
        #print("!",a)
    print("k =", k)
    print("test accuracy:", totalTest / 10)
    print("******************************")

