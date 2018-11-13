# Tool
from heapq import *

import matplotlib.cm as cm  # Color display
import matplotlib.pyplot as plt
# ML
import numpy as np
import pandas as pd
import scipy.io
import scipy.misc
from sklearn.model_selection import train_test_split

np.random.seed(1)

'''
# 中位数循环拆分kd树
class TreeNode:
    def __init__(self, data, val, pos, len):
        self.data = data
        self.val = val
        self.pos = pos
        self.len = len
        # self.neighbour = neighbour
        self.left = None
        self.right = None


# neighbourIdShape, 2dir, 784dimension
neighbour = [[None for _ in range(2)] for l in range(784)]


# X_tree_train idx 0: label idx1-784:784pixel
def kdTree(X):
    if len(X) == 1:
        return TreeNode(X, X[0][0], -1, 1)
    Q = X.T
    max_var = np.var(Q[1])
    max_idx = 1
    for i in range(2, len(Q)):
        if np.var(Q[i]) > max_var:
            max_idx = i
            max_var = np.var(Q[i])
    mid = np.median(Q[max_idx])
    cur = TreeNode(X, mid, max_idx, len(X))
    left_idx = []
    right_idx = []
    if mid == 0:
        for i in range(len(X)):
            if Q[max_idx][i] == 0:
                left_idx.append(i)
            else:
                right_idx.append(i)
    else:
        for i in range(len(X)):
            if Q[max_idx][i] < mid:
                left_idx.append(i)
            else:
                right_idx.append(i)
    if left_idx:
        # neighbourId += 1
        left_subX = X[[left_idx]][:]
        cur.left = kdTree(left_subX)
    if right_idx:
        right_subX = X[[right_idx]][:]
        cur.right = kdTree(right_subX)
    return cur
'''

'''
PCA part
'''


# PCA 构造出新的单位向量组*向量组对应的系数*向量组对应的转置
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


def recoverData(Z, U, K):
    Ureduced = U[:, :K]
    Xapprox = Z.dot(Ureduced.T)
    return Xapprox

'''

def displayData(X, mynrows=10, myncols=10, id=0):
    width, height = 28, 28
    nrows, ncols = mynrows, myncols
    big_picture = np.zeros((height * nrows, width * ncols))
    irow, icol = 0, 0
    for idx in range(nrows * ncols):
        if icol == ncols:
            irow += 1
            icol = 0
        iimg = getDatumImg(X[idx])
        # img = scipy.misc.toimage(iimg)
        # plt.imshow(img, cmap=cm.Greys_r)
        # plt.show()
        big_picture[irow * height:irow * height + iimg.shape[0], icol * width:icol * width + iimg.shape[1]] = iimg
        icol += 1
    
    #fig = plt.figure(figsize=(10, 10))
    #img = Image.fromarray(big_picture * 255)
    #plt.imshow(img, cmap=cm.Greys_r)
    

    fig = plt.figure(figsize=(10, 10))
    img = scipy.misc.toimage(big_picture)
    # csvT = pd.DataFrame(big_picture)
    # csvT.to_csv('../ML/data/CS6364HW1digit/big_picture' + str(id) + '.csv')
    plt.imshow(img, cmap=cm.Greys_r)
    plt.show()


def getDatumImg(row):
    width, height = 28, 28
    square = row.reshape(width, height)
    return square.T
'''

'''
train = pd.read_csv('../ML/data/CS6364HW1digit/train.csv')
test = pd.read_csv('../ML/data/CS6364HW1digit/test.csv')
X_tree_train = train.values
np.save('../ML/data/CS6364HW1digit/X_tree_train_KNN',X_tree_train)

# first select ten number as centroid
# I should use numpy
# 783 groups of centroids, each group have 10 centroids and each centroid in group is label
# size = 10 & 783. add opperation: (preval * pretimes + curval)/(pretimes+1) find nearest operation: min(pre - cur)

Y_train = train["label"].values
X_train = train.drop(labels=["label"], axis=1).values

del train

# show digit count
# g = sns.countplot(Y_train)

# plt.show()
# print(Y_train.value_counts())

# normalization
X_train = X_train /255
test = test/255

np.save('../ML/data/CS6364HW1digit/X_train_KNN',X_train)
np.save('../ML/data/CS6364HW1digit/Y_train_KNN',Y_train)
np.save('../ML/data/CS6364HW1digit/test_KNN',test)


X_train = np.load('../ML/data/CS6364HW1digit/X_train_KNN.npy')
Y_train = np.load('../ML/data/CS6364HW1digit/Y_train_KNN.npy')

# split the train set to get validation set
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train,Y_train,test_size = 0.1, random_state=1)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_validation,Y_validation,test_size=0.5,random_state=1)
np.save('../ML/data/CS6364HW1digit/X_validation_KNN',X_validation)
np.save('../ML/data/CS6364HW1digit/Y_validation_KNN',Y_validation)
np.save('../ML/data/CS6364HW1digit/X_test_KNN',X_test)
np.save('../ML/data/CS6364HW1digit/Y_test_KNN',Y_test)


# find digit appearing times in Y_train
digit_train = [0] * 10
for val in Y_train:
    digit_train[val] += 1
print(digit_train)
'''


X_tree_train = np.load('../ML/data/CS6364HW1digit/X_tree_train_KNN.npy')
# construct tree is time cost, so we take 1% of train part as test data
X_tree_train, X_ref = train_test_split(X_tree_train, test_size=0.99, random_state=1)
X_train = np.load('../ML/data/CS6364HW1digit/X_train_KNN.npy')
Y_train = np.load('../ML/data/CS6364HW1digit/Y_train_KNN.npy')
# X_validation = np.load('../ML/data/CS6364HW1digit/X_validation_KNN.npy')
# Y_validation = np.load('../ML/data/CS6364HW1digit/Y_validation_KNN.npy')
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)

# X_test = np.load('../ML/data/CS6364HW1digit/X_test_KNN.npy')
# Y_test = np.load('../ML/data/CS6364HW1digit/Y_test_KNN.npy')
featureSize = len(X_train[0])
# X_train,X_ref,Y_train,Y_ref = train_test_split(X_train,Y_train,test_size = 0.99, random_state=1)
# X_validation,X_ref,Y_validation,Y_ref = train_test_split(X_validation,Y_validation,test_size = 0.98, random_state=1)
# primogenitor = kdTree(X_tree_train)


# parts of test.csv prediction generation

train = pd.read_csv('../ML/data/CS6364HW1digit/train.csv')
Y_train = train["label"].values
X_train = train.drop(labels=["label"], axis=1).values
test = pd.read_csv('../ML/data/CS6364HW1digit/test.csv')
test = test.values
# attetion, Z_validation should be changed to (test,U,K= 40)


# displayData(X_validation)
'''
PCA part2
'''

U, S, V = getUSV(X_train)
Z_train = projectData(X_train, U, K=50)
Z_validation = projectData(test, U, K=50)
# displayData(U[:, :50].T, mynrows=7, myncols=7,id = 2)
K = recoverData(Z_validation, U, K=50)
# displayData(K)

print(U.shape)
print(S.shape)
print(V.shape)
print(np.sum(np.square(V),axis=-1))
print(Z_train.shape)

# root = kdTree(X_train,0,len(X_train[0]))

'''
In this part we can find, many points are meanningless, totally 228 meanningless point.

Q = X_train.T
useful = []
for i in range(len(Q)):
    # print(i,sum(Q[i])/len(Q[i]))
    if max(Q[i]) != 0:
        useful.append(i)
# print(count,len(count))
Q = Q[useful]
# X_train = Q.T
'''

'''
************************************************************************************
'''


def distance(p1, p2):
    # p2 = p2.T[useful].T
    return np.sum(np.square(p1 - p2), axis=-1)


confusionMatrix = [[0 for i in range(10)] for j in range(10)]


# KNN初始测试
# 不需要训练 forloop multi
def validation(X, Y, X_val, Y_val, k_val, f_Size, l_size):
    correct = 0
    val_total = len(X_val)
    train_total = len(X)
    for v_idx in range(val_total):
        heap = []
        count = 0
        balance = [0] * l_size
        predict_Y = 0
        for t_idx in range(train_total):
            curDistance = 0
            for f_idx in range(f_Size):
                curDistance -= ((X[t_idx][f_idx] - X_val[v_idx][f_idx])) ** 2
            if count == k_val:
                if curDistance > heap[0][0]:
                    balance[heappop(heap)[1]] -= 1
                    balance[Y[t_idx]] += 1
                    heappush(heap, (curDistance, Y[t_idx]))
            else:
                count += 1
                balance[Y[t_idx]] += 1
                heappush(heap, (curDistance, Y[t_idx]))
            # print("***************")
        cur = 0
        maxCur = balance[0]
        for i in range(1, l_size):
            if balance[i] > maxCur:
                maxCur = balance[i]
                cur = i
        if cur == Y_val[v_idx]:
            correct += 1

    return correct / val_total


# vector multi
def validation2(X, Y, X_val, Y_val, k_val, f_size, l_size):
    correct = 0
    for v_idx in range(len(X_val)):
        heap = []
        balance = [0] * l_size
        dist = distance(X, X_val[v_idx])
        count = 0
        for t_idx in range(len(X)):
            cur_sum = -dist[t_idx]
            if count == k_val:
                if cur_sum > heap[0][0]:
                    balance[heappop(heap)[1]] -= 1
                    balance[Y[t_idx]] += 1
                    heappush(heap, (cur_sum, Y[t_idx]))
            else:
                count += 1
                balance[Y[t_idx]] += 1
                heappush(heap, (cur_sum, Y[t_idx]))
        cur = 0
        maxCur = balance[0]
        for i in range(1, l_size):
            if balance[i] > maxCur:
                maxCur = balance[i]
                cur = i
        # print(cur,Y_val[v_idx])

        if cur == Y_val[v_idx]:
            correct += 1

    return correct / len(Y_val)


# used to generate predict result
def validation2GenerateMode(X, Y, X_val, k_val, l_size):
    res = []
    for v_idx in range(len(X_val)):
        heap = []
        balance = [0] * l_size
        dist = distance(X, X_val[v_idx])
        count = 0
        for t_idx in range(len(X)):
            cur_sum = -dist[t_idx]
            if count == k_val:
                if cur_sum > heap[0][0]:
                    balance[heappop(heap)[1]] -= 1
                    balance[Y[t_idx]] += 1
                    heappush(heap, (cur_sum, Y[t_idx]))
            else:
                count += 1
                balance[Y[t_idx]] += 1
                heappush(heap, (cur_sum, Y[t_idx]))
        cur = 0
        maxCur = balance[0]
        for i in range(1, l_size):
            if balance[i] > maxCur:
                maxCur = balance[i]
                cur = i
        res.append(cur)
        #print(cur,v_idx)
    finalRes = np.asarray(res)
    df = pd.DataFrame(finalRes)
    df.index = np.arange(1, len(df) + 1)
    df.to_csv("../ML/data/CS6364HW1digit/submission.csv", header=["Label"], index_label=["ImageId"])
    pd.DataFrame.to_csv()


def validation3(X_val, Y_val, k_val, root):
    def searchKdTree(node, val):
        if node.left and node.right:
            if node.left.len < k_val and node.right.len < k_val:
                return node
        else:
            return node
        if val[node.pos] < node.val:
            return searchKdTree(node.left, val)
        else:
            return searchKdTree(node.right, val)

    correct = 0
    for v_idx in range(len(X_val)):
        min_train = searchKdTree(root, X_val[v_idx]).data
        heap = []
        balance = [0] * len(min_train)
        dist = distance(min_train[:, 1:], X_val[v_idx])
        count = 0
        for t_idx in range(len(min_train)):
            cur_sum = -dist[t_idx]
            if count == k_val:
                if cur_sum > heap[0][0]:
                    balance[heappop(heap)[1]] -= 1
                    balance[min_train[t_idx][0]] += 1
                    heappush(heap, (cur_sum, min_train[t_idx][0]))
            else:
                count += 1
                balance[min_train[t_idx][0]] += 1
                heappush(heap, (cur_sum, min_train[t_idx][0]))
        cur = 0
        maxCur = balance[0]
        for i in range(1, len(min_train)):
            if balance[i] > maxCur:
                maxCur = balance[i]
                cur = i
        # print(cur,Y_val[v_idx])
        if cur == Y_val[v_idx]:
            correct += 1
    return correct / len(Y_val)


print(validation2(X_train,Y_train,X_validation[:10],Y_validation[:10],15,featureSize,10))
# print(validation2(Z_train,Y_train,Z_validation,Y_validation,10,featureSize,10))
# print(validation3(X_validation, Y_validation, 60, primogenitor))
# print(validation2GenerateMode(Z_train, Y_train, Z_validation, 10, 10))
if 1 == 0:
    print("Principle Component Take: first 50")
    print("420 validation data")
    for k in range(6, 16, 2):
        print("Accuracy:", validation2(Z_train, Y_train, Z_validation, Y_validation, k, featureSize, 10), "k:", k)

