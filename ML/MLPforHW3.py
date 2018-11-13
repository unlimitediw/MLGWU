import sys
import itertools

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.io
from scipy.special import expit  # Vectorized sigmoid function
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

sys.path.append("..")
from ML import EntropyGainGenerator
from sklearn import svm
from ML import CS6364_HW3_SVM_Handwork as SH
from ML import KFoldValidation as KF
import matplotlib.pyplot as plt
np.random.seed(1)


# data read
datafile = "/Users/unlimitediw/PycharmProjects/MLGWU/ML/data/CS6364HW3Adult/adult.csv"
data = pd.read_csv(datafile)

# drop ? elements
print('DataLenth:', len(data))
data = data[data.occupation != '?']
keys = data.keys()
for key in keys:
    if data[key].dtype != 'int64':
        data = data[data[key] != '?']
df = pd.DataFrame(data)
df.to_csv("/Users/unlimitediw/PycharmProjects/MLGWU/ML/data/CS6364HW3Adult/treatedAdult.csv", index=False)

# X Y data initialization
df = pd.read_csv("/Users/unlimitediw/PycharmProjects/MLGWU/ML/data/CS6364HW3Adult/treatedAdult.csv")
Y = df["income"].values
X = df.drop(labels=["income"], axis=1).values

print(X.shape,Y.shape)
# random set data

'''
indices = np.random.choice(X.shape[0], X.shape[0])
X = X[indices]
Y = Y[indices]
'''

# feature classification
X = X.T

# X
'''
# 0 age: range from 17 to 90
# 1 workclass: {'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay', 'Self-emp-not-inc', 'State-gov', 'Private'}
# 2 fnlwgt: the number of people the census takers believe that observation represents
# 3 education: Preschool < 1st-4th < 5th-6th < 7th-8th < 9th < 10th < 11th < 12th < HS-grad < Prof-school < Assoc-acdm < Assoc-voc < Some-college < Bachelors < Masters < Doctorate
# 4 education num: Highest level of education in numerical form
# 5 marital status: {'Separated', 'Never-married', 'Married-AF-spouse', 'Divorced', 'Married-spouse-absent', 'Widowed', 'Married-civ-spouse'}
# 6 occupation: {'Tech-support', 'Sales', 'Adm-clerical', 'Transport-moving', 'Prof-specialty', 'Protective-serv', 'Exec-managerial', 'Priv-house-serv', 'Craft-repair', 'Farming-fishing', 'Handlers-cleaners', 'Armed-Forces', 'Other-service', 'Machine-op-inspct'}
# 7 relationship: {'Other-relative', 'Husband', 'Not-in-family', 'Unmarried', 'Wife', 'Own-child'}
# 8 race: {'White', 'Other', 'Black', 'Amer-Indian-Eskimo', 'Asian-Pac-Islander'}
# 9 sex: {'Male', 'Female'}
# 10 capital gain: income from investment sources, apart from wages
# 11 capital loss: ~
# 12 hours.per.week
# 13 native country: {'Ecuador', 'Yugoslavia', 'Trinadad&Tobago', 'India', 'Honduras', 'Japan', 'France', 'Holand-Netherlands', 'Hong', 'Columbia', 'Puerto-Rico', 'Jamaica', 'Scotland', 'Mexico', 'United-States', 'Iran', 'South', 'Peru', 'El-Salvador', 'Haiti', 'Portugal', 'Italy', 'Dominican-Republic', 'Canada', 'Poland', 'Philippines', 'Germany', 'Cuba', 'China', 'Guatemala', 'Ireland', 'Cambodia', 'Vietnam', 'Greece', 'Thailand', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Laos', 'England', 'Taiwan', 'Hungary'}
'''

# discrete data to integers
# Y
Y_0 = Y == "<=50K"
Y_1 = Y == ">50K"
Y[Y_0] = 0
Y[Y_1] = 1

data = df.values


from sklearn.model_selection import train_test_split

X = X[[0,2,4,10,11,12], :]
X = X.T
Y = Y.astype('int')
X = np.float64(X)

def standarize(X):
    X = X.T
    for i in range(1,len(X)):
        mean = np.mean(X[i])
        std = np.std(X[i])
        X[i] = (X[i] - mean) / std
    X = X.T
standarize(X)
X, X_show, Y, Y_show = train_test_split(X, Y, test_size=0.97, random_state=15)
pos = np.asarray([X_show[t] for t in range(X_show.shape[0]) if Y_show[t] == 1])
neg = np.asarray([X_show[t] for t in range(X_show.shape[0]) if Y_show[t] == -1])

input_layer_size = 6  # plus 1 for X0
hidden_layer_size = 3  # plus 1 for X0
output_layer_size = 1  # digits

# First step: insert 1 to X
# Second step: construct a 784,25 shape random set matrix
# Third step: construct a 26 * 10 shape random set matrix
# Fourth step: do the forward propagation
# calculate the lost at the end and do backward propagation

X = np.insert(X, 0, 1, axis=1)

X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

m = len(X)
# theta initialization
def genRandThetas():
    epsilon = 0.12
    # attention, theta is at left
    theta1_shape = (hidden_layer_size, input_layer_size + 1)
    theta2_shape = (output_layer_size, hidden_layer_size + 1)
    return np.random.rand(*theta1_shape) * 2 * epsilon - epsilon, np.random.rand(*theta2_shape) * 2 * epsilon - epsilon

# if we need to use fmin_cg, we should use flattenParams and reshapeParams
def flattenParams(Thetas):
    flattened_list = [mytheta.flatten() for mytheta in Thetas]
    combined = list(itertools.chain.from_iterable(flattened_list))
    assert len(combined) == (input_layer_size+1)*hidden_layer_size + \
                            (hidden_layer_size+1)*output_layer_size
    return np.array(combined).reshape((len(combined), 1))


def reshapeParams(flattened_list):
    theta1 = flattened_list[:(input_layer_size + 1) * hidden_layer_size].reshape(
        (hidden_layer_size, input_layer_size + 1))
    theta2 = flattened_list[(input_layer_size + 1) * hidden_layer_size:].reshape(
        (output_layer_size, hidden_layer_size + 1))

    return [theta1,theta2]

def flattenX(X):
    train_size = len(X)
    return np.array(X.flatten()).reshape((train_size*(input_layer_size+1),1))

def reshapeX(X,preSize):
    return np.array(X).reshape((preSize,input_layer_size+1))


# Feedforward
def propagateForward(X, Thetas):
    # Thetas = [theta1, theta2]
    features = X
    z_memo = []
    for i in range(len(Thetas)):
        theta = Thetas[i]
        z = theta.dot(features).reshape((theta.shape[0], 1))
        # activation should be 0-0.5, 0.5-1
        a = expit(z)
        z_memo.append((z, a))
        if i == len(Thetas) - 1:
            return np.array(z_memo)
        a = np.insert(a, 0, 1)  # add X0
        features = a

def computeCost(Thetas, X, Y, my_lambda=0.):
    Thetas = reshapeParams(Thetas)
    X = reshapeX(X,m)
    total_cost = 0.
    train_size = m
    try:
        for i in range(train_size):
            cur_X = X[i]
            cur_Y = Y[i]
            hyper = propagateForward(cur_X.T, Thetas)[-1][1]
            cost = - (cur_Y * np.log(hyper)) - (1 - cur_Y)*(np.log(1 - hyper))
            total_cost += cost
    except:
        print("train_size should be smaller than X size")
    total_cost = float(total_cost) / train_size

    # avoid overfitting
    total_reg = 0.
    for theta in Thetas:
        total_reg += np.sum(theta*theta)
    total_reg *= float(my_lambda) / (2 * train_size)
    return total_cost + total_reg


# Backpropagation part
def sigmoidGradient(z):
    # expit = 1/(1+e^z)
    # dummy is the activation layer
    dummy = expit(z)
    return dummy * (1 - dummy)


def backPropagate(Thetas, X, Y, my_lambda=0.):
    Thetas = reshapeParams(Thetas) # ccc
    X = reshapeX(X,m)
    train_size = m
    Delta1 = np.zeros((hidden_layer_size, input_layer_size + 1))
    Delta2 = np.zeros((output_layer_size, hidden_layer_size + 1))
    for i in range(train_size):
        cur_X = X[i]
        cur_Y = Y[i]
        a1 = cur_X.reshape((input_layer_size + 1, 1))
        temp = propagateForward(cur_X, Thetas)
        z2 = temp[0][0]
        a2 = temp[0][1]
        z3 = temp[1][0]
        a3 = temp[1][1]
        # delta is just a diff, Delta is gradient
        delta3 = a3 - cur_Y
        # bp should remove first X0
        # Thetas[1].T[1:,:].dot(delta3) is the theta.dot(pre_error)
        delta2 = Thetas[1].T[1:, :].dot(delta3) * sigmoidGradient(z2)
        a2 = np.insert(a2, 0, 1, axis=0)
        # Delta = (antriTri + regular)/size
        # delat.dot(activation) + pre = new antiTri
        # print(delta3.shape,a2.T.shape,Delta2.shape)
        Delta1 += delta2.dot(a1.T)
        Delta2 += delta3.dot(a2.T)
        # Finally Delta = derivative of theta for each layer

    D1 = Delta1 / train_size
    D2 = Delta2 / train_size

    # Regularization:
    D1[:, 1:] += (my_lambda / train_size) * Thetas[0][:, 1:]
    D2[:, 1:] += (my_lambda / train_size) * Thetas[1][:, 1:]

    return flattenParams([D1, D2]).flatten()



def trainNN2(X, Y, my_lambda=0.):
    theta1, theta2 = genRandThetas()
    f_theta = flattenParams([theta1,theta2])
    result = scipy.optimize.fmin_cg(computeCost, x0=f_theta, fprime=backPropagate, args=(flattenX(X), Y, my_lambda),
                                    maxiter=30, disp=True, full_output=True)
    return reshapeParams(result[0])


# argmax for softmax
def predictNN(X, Thetas):
    output = propagateForward(X, Thetas)[-1][1]
    return 1 if output > 0.5 else 0


def computeAccuracy(Thetas, X, Y):
    correct = 0
    total = X.shape[0]
    print(X.shape)
    for i in range(total):
        hyper = predictNN(X[i], Thetas)
        print(hyper,Y[i])
        if hyper == Y[i]:
            correct += 1
    return "%0.1f%%" % (100 * correct / total)
# Predict validating and testing
learn = trainNN2(X,Y)
print(computeAccuracy(learn,X,Y))