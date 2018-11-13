# ML
# Tools
import itertools

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.io
from scipy.special import expit  # Vectorized sigmoid function
from sklearn.model_selection import train_test_split

np.random.seed(1)



train = scipy.io.loadmat("../ML/data/229/ex4data1.mat")
# MNIST dataset
train = pd.read_csv('../ML/data/CS6364HW1digit/train.csv')
Y_train = train["label"].values
X_train = train.drop(labels=["label"], axis=1).values


indice = np.random.choice(len(X_train),len(X_train),replace = False)
X_train = X_train[indice]
Y_train = Y_train[indice]
# 229 dataset
#X_train = train['X']
#Y_train = train['y']

# first layer 785 * 25
# second layer 26 * 10
input_layer_size = 784  # plus 1 for X0
hidden_layer_size = 35  # plus 1 for X0
output_layer_size = 10  # digits

# First step: insert 1 to X_train
# Second step: construct a 784,25 shape random set matrix
# Third step: construct a 26 * 10 shape random set matrix
# Fourth step: do the forward propagation
# calculate the lost at the end and do backward propagation

X_train = np.insert(X_train, 0, 1, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=1)


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
    X = reshapeX(X,len(X_train))
    total_cost = 0.
    train_size = len(X_train)
    try:
        for i in range(train_size):
            cur_X = X[i]
            cur_Y = Y[i]
            hyper = propagateForward(cur_X.T, Thetas)[-1][1]
            temp_Y = np.zeros((10, 1))
            temp_Y[cur_Y - 1] = 1
            cost = - temp_Y.T.dot(np.log(hyper)) - (1 - temp_Y.T).dot(np.log(1 - hyper))
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
    X = reshapeX(X,len(X_train))
    train_size = len(X_train)
    Delta1 = np.zeros((hidden_layer_size, input_layer_size + 1))
    Delta2 = np.zeros((output_layer_size, hidden_layer_size + 1))
    for i in range(train_size):
        cur_X = X[i]
        a1 = cur_X.reshape((input_layer_size + 1, 1))
        temp = propagateForward(cur_X, Thetas)
        z2 = temp[0][0]
        a2 = temp[0][1]
        z3 = temp[1][0]
        a3 = temp[1][1]
        temp_Y = np.zeros((10, 1))
        temp_Y[Y[i] - 1] = 1
        # delta is just a diff, Delta is gradient
        delta3 = a3 - temp_Y
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


def trainNN(lr, my_lambda=0.):
    theta1, theta2 = genRandThetas()
    for _ in range(2):
        D1, D2 = reshapeParams(backPropagate([theta1, theta2], X_train, Y_train, my_lambda))
        theta1 -= lr * D1
        theta2 -= lr * D2
    return [theta1, theta2]


def trainNN2(X, Y, my_lambda=0.):
    theta1, theta2 = genRandThetas()
    f_theta = flattenParams([theta1,theta2])
    result = scipy.optimize.fmin_cg(computeCost, x0=f_theta, fprime=backPropagate, args=(flattenX(X), Y, my_lambda),
                                    maxiter=50, disp=True, full_output=True)
    return reshapeParams(result[0])


# argmax for softmax
def predictNN(X, Thetas):
    classes = list(range(1, 10)) + [10]
    output = propagateForward(X, Thetas)
    return classes[np.argmax(output[-1][1])]


def computeAccuracy(Thetas, X, Y):
    correct = 0
    total = X.shape[0]
    for i in range(total):
        if int(predictNN(X[i], Thetas) == Y[i]):
            correct += 1
    return "%0.1f%%" % (100 * correct / total)

print(X_train.shape,Y_train.shape)
learn = trainNN2(X_train,Y_train)
#learn = trainNN(1,2)
print(computeAccuracy(learn, X_test, Y_test))
print(computeCost(flattenParams(learn),flattenX(X_train),Y_train))


# def trainNN(mylambda = 0.)

# print(computeCost([theta1, theta2], X_train, Y_train, 10))

