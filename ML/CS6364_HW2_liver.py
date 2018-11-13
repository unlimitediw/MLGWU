import numpy as np
import scipy.io.arff
import pandas as pd
from scipy.special import expit

from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
# Display Y_label
# plt.bar([0,1],[len(PIMA_Y_train)-sum(PIMA_Y_train),sum(PIMA_Y_train)])
# plt.xlabel("outcome")
# plt.ylabel("count")
# plt.title("PIMA outcome count")
# plt.xticks(np.arange(0, 2, 1))
# plt.show()


np.random.seed(0)

'''
# algorithm part
'''
# check labels
label_set = [0.0, 0.5, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 16.0, 20.0]
label_dic = {}
predict_dic = {}
for i in range(len(label_set)):
    label_dic[label_set[i]] = i
    predict_dic[i] = label_set[i]

# loss visualization
iterX = []
lossY = []

def train(Thetas, X, Y, lr, loops):
    f = len(X[0])
    for l in range(loops):
        Delta = np.zeros((16, f))
        for i in range(len(X)):
            predict = expit(Thetas.dot(X[i].T)).reshape((16, 1))
            Y_set = np.zeros((16, 1))
            Y_set[label_dic[Y[i][0]]] = 1
            # print(label_dic[Y[i][0]],Y[i][0])
            delta = (predict - Y_set) * abs(predict - Y_set)
            Delta += delta.dot(X[i].T.reshape(1, f))
        bias = 8000 / len(X) * Thetas
        Delta /= len(X)
        iterX.append(l)
        lossY.append(np.sum(np.square(Delta)))
        Thetas -= lr * (Delta + bias)
        #print("iterTimes:",l,"Thetas:",Thetas[0])

# confidence matric
totalList = []
truePositiveList = []
falsePositiveList = []
trueNegativeList = []
falseNegativeList = []
TPRList = []
TNRList = []


def validation(Thetas, X, Y):
    correct = 0
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    total = len(X)
    for i in range(len(X)):
        res = expit(Thetas.dot(X[i].T).reshape(16, 1))
        # print(res)
        predict = np.argmax(res)
        # print(predict,label_dic[Y[i][0]])
        '''
        # accurate validation
        if predict == label_dic[Y[i][0]]:
            correct += 1
        '''
        if (predict <= 5 and label_dic[Y[i][0]] <= 5) or (predict > 5 and label_dic[Y[i][0]] > 5):
            correct += 1
        # for confidence metric
        if predict <= 5 and label_dic[Y[i][0]] <= 5:
            trueNegative += 1
        if predict > 5 and label_dic[Y[i][0]] > 5:
            truePositive += 1
        if predict <= 5 and label_dic[Y[i][0]] > 5:
            falseNegative += 1
        if predict > 5 and label_dic[Y[i][0]] <= 5:
            falsePositive += 1
    # for confidence metric
    TPR = format(truePositive/ (truePositive + falsePositive),'.03f') if (truePositive + falsePositive) != 0 else None
    TNR = format(trueNegative / (trueNegative + falsePositive),'.03f') if (trueNegative + falsePositive) != 0 else None
    totalList.append(total)
    truePositiveList.append(truePositive)
    falsePositiveList.append(falsePositive)
    trueNegativeList.append(trueNegative)
    falseNegativeList.append(falseNegative)
    TPRList.append(TPR)
    TNRList.append(TNR)
    return correct / len(X)

def train2(Thetas, X, Y, lr, loops):
    X_length = len(X)
    for l in range(loops):
        for i in range(X_length):
            predict = Thetas.dot(X[i].T)
            # rint(predict)
            err = label_dic[Y[i][0]] - predict
            # print(err)
            # print(err)
            Thetas += lr * X[i] * err


def validation2(Thetas, X, Y):
    correct = 0
    for i in range(len(X)):
        predict = Thetas.dot(X[i].T)
        '''
        # accurate validation
        if abs(predict - label_dic[Y[i][0]]) < 0.5:
            correct += 1
        '''
        if (predict <= 5 and label_dic[Y[i][0]] <= 5) or (predict > 5 and label_dic[Y[i][0]] > 5):
            correct += 1
    return correct / len(X)


def train3(Thetas, X, Y, lr, loops):
    X_length = len(X)
    for l in range(loops):
        for i in range(X_length):
            predict = 1 if expit(Thetas.dot(X[i].T)) > 0.5 else 0
            actual_Y = 1 if Y[i] <= 5 else 0
            err = actual_Y - predict
            Thetas += lr * X[i] * err


def validation3(Thetas, X, Y):
    correct = 0
    for i in range(len(X)):
        predict = 1 if expit(Thetas.dot(X[i].T)) > 0.5 else 0
        actual_Y = 1 if Y[i] <= 5 else 0
        if predict == actual_Y:
            correct += 1
    return correct / len(X)


def train4(Thetas, X, Y, lr, loops):
    X_length = len(X)
    for l in range(loops):
        for i in range(X_length):
            predict = 1 if expit(Thetas.dot(X[i].T)) > 0.5 else 0
            actual_Y = Y[i]
            err = actual_Y - predict
            Thetas += lr * X[i] * err


def validation4(Thetas, X, Y):
    correct = 0
    for i in range(len(X)):
        predict = 1 if expit(Thetas.dot(X[i].T)) > 0.5 else 0
        if predict == Y[i]:
            correct += 1
    return correct / len(X)


'''
# data preprocessing, train and validation
'''

# train = pd.read_csv("../ML/data/CS6364HW2Liver/dataset_8_liver-disorders.csv").values
origin = scipy.io.arff.loadarff("../ML/data/CS6364HW2Liver/dataset_8_liver-disorders.arff")[0]

X_train = []
X_test = []
Y_train = []
Y_test = []

# random 9:1 train:test spilt
treated = []
treated7 = []
treated7_y = []

for i in range(len(origin)):
    origin[i][6].astype(float)
for i in range(len(origin)):
    treated.append(list(origin[i])[:6])
    treated7.append(list(origin[i])[:5])
    treated7_y.append(int(origin[i][6]) - 1)
treated = np.asarray(treated)
treated = np.insert(treated, 0, 1, axis=1)
treated7 = np.asarray(treated7)
treated7 = np.insert(treated7, 0, 1, axis=1)
sevenLen = len(treated7[0])
sixLen = len(treated[0])
treated7_y = np.asarray(treated7_y)

# X_treated = treated[:, :6]
# Y_treated = treated[:, 6:]

# normalization
treated = treated.T
for i in range(1, sixLen - 1):
    # std scope
    treated[i] = (treated[i] - np.mean(treated[i])) / np.std(treated[i])
    # minmax scope
    # X_treated[i] = (X_treated[i] - np.min(X_treated[i]))/(np.max(X_treated[i])- np.min(X_treated[i]))
    # mean scope
    # X_treated[i] /= np.mean(X_treated[i])
treated = treated.T

treated7 = treated7.T
for i in range(1, sevenLen):
    treated7[i] = (treated7[i] - np.mean(treated7[i])) / np.std(treated7[i])
treated7 = treated7.T

acc = 0
s_times = 50
iter_times = 200
kf = KFold(n_splits=s_times)
switch = 3
if switch == 1:
    for cur_train, cur_test in kf.split(treated):
        # softMax way
        Thetas = np.random.rand(16, sixLen - 1)
        # regression way and logistic way
        # Thetas = np.random.rand(6)
        # random spilt
        # X_train, X_test, Y_train, Y_test = train_test_split(X_treated, Y_treated, test_size=0.1, random_state=i)
        # cross validation spilt process
        X_train = treated[cur_train][:, :sixLen - 1]
        X_test = treated[cur_test][:, :sixLen - 1]
        Y_train = treated[cur_train][:, sixLen - 1:]
        Y_test = treated[cur_test][:, sixLen - 1:]
        train(Thetas, X_train, Y_train, 0.05, iter_times)
        cur_acc = validation(Thetas, X_test, Y_test)
        acc += cur_acc
elif switch == 2:
    for i in range(s_times):
        # softMax way
        # Thetas = np.random.rand(16, 6)
        # regression way and logistic way
        Thetas = np.random.rand(sixLen - 1)
        # random spilt
        cur_train, cur_test = train_test_split(treated, test_size=0.1, random_state=i)
        # cross validation spilt process
        X_train = cur_train[:, :sixLen - 1]
        X_test = cur_test[:, :sixLen - 1]
        Y_train = cur_train[:, sixLen - 1:]
        Y_test = cur_test[:, sixLen - 1:]
        train2(Thetas, X_train, Y_train, 0.005, iter_times)
        cur_acc = validation2(Thetas, X_test, Y_test)
        acc += cur_acc
        print("random validation acc for", i, "times:", cur_acc)
elif switch == 3:
    # treated = np.insert(treated, 0, 1, axis=1)
    sixLen -= 1
    for i in range(s_times):
        #Thetas = np.random.rand(7)
        Thetas = np.random.rand(16, sixLen)
        cur_train, cur_test = train_test_split(treated, test_size=0.1, random_state=i)
        X_train = cur_train[:, :sixLen]
        X_test = cur_test[:, :sixLen]
        Y_train = cur_train[:, sixLen:]
        Y_test = cur_test[:, sixLen:]
        train(Thetas, X_train, Y_train, 0.005, iter_times)
        cur_acc = validation(Thetas, X_test, Y_test)
        acc += cur_acc
        print("random validation acc for", i, "times:", cur_acc)
    final = np.asarray([np.asarray(totalList), np.asarray(truePositiveList), np.asarray(trueNegativeList),
                        np.asarray(falsePositiveList), np.asarray(falseNegativeList), np.asarray(TPRList),
                        np.asarray(TNRList)])
    df = pd.DataFrame(final.T)
    print(df)
    df.index = np.arange(1, len(df) + 1)
    df.to_csv("../ML/data/CS6364HW2Liver/confidenceMetric.csv",
              index_label=["iteration times", "total test", "truePositive", "trueNegative", "falsePositive",
                           "falseNegative", "TPR", "TNR"])
else:
    for i in range(s_times):
        Thetas = np.random.rand(sevenLen)
        cur_train, cur_test = train_test_split(treated7, test_size=0.1, random_state=i)
        cur_train_y, cur_test_y = train_test_split(treated7_y, test_size=0.1, random_state=i)
        X_train = cur_train
        X_test = cur_test
        Y_train = cur_train_y
        Y_test = cur_test_y
        train4(Thetas, X_train, Y_train, 0.03, iter_times)
        cur_acc = validation4(Thetas, X_test, Y_test)
        acc += cur_acc
        print("random validation acc for", i, "times:", cur_acc)

plt.plot(iterX,lossY)
plt.xlabel("Times of iteration")
plt.ylabel("Square error sum for 16 * 6 size weight")
plt.title("Relationship between times of iteration and model training loss")
plt.show()
print("acc", acc / s_times)

ax = plt.figure(figsize=(9,9))
for i in range(len(TNRList)):
    TPRList[i] = float(TPRList[i])
    TNRList[i] = float(TNRList[i])
x = [i for i in range(1,s_times+1)]
plt.ylim([0.0,1.0])
plt.plot(x,TPRList,"ro",label = "TPR, Positive correct rate")
plt.plot(x,TNRList,"bo",label = "TNR, Negative correct rate")
plt.xlabel("random seed")
plt.ylabel("prediction correct rate")
plt.title("relationship between Positive and Negative prediction")
plt.legend()
plt.show()
print("acc", acc / s_times)
