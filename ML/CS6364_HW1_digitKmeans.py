import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

np.random.seed(1)
'''
digit rec 10-NN 783 features
假设只有一个点 随机抽取10个点作为初始点，然后进行分类和标记
以此推广，并对每个点进行权重配置 灰度越高权重
'''
from sklearn.model_selection import train_test_split # get validation set
import itertools


np.random.seed(1)
'''
# Some data treatment methods are learned from @YassineGhouzam

# digit rec 10-NN 783 features
# 假设只有一个点 随机抽取10个点作为初始点，然后进行分类和标记
# 以此推广，并对每个点进行权重配置 灰度越高权重


train = pd.read_csv('../ML/data/CS6364HW1digit/train.csv')
test = pd.read_csv('../ML/data/CS6364HW1digit/test.csv')

# first select ten number as centroid
# I should use numpy
# 783 groups of centroids, each group have 10 centroids and each centroid in group is label
# size = 10 & 783. add opperation: (preval * pretimes + curval)/(pretimes+1) find nearest operation: min(pre - cur)

Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

del train

# show digit count
# g = sns.countplot(Y_train)

# plt.show()
# print(Y_train.value_counts())

# normalization
X_train = X_train /255
test = test/255

# reshape all data to 28X28X(1) 3D matrices
X_train = X_train.values.reshape(-1,28,28) # change to numpy.ndarray
Y_train = Y_train.values
test = test.values.reshape(-1,28,28) # -1 means default slice

np.save('../ML/data/CS6364HW1digit/X_train_Kmeans',X_train)
np.save('../ML/data/CS6364HW1digit/Y_train_Kmeans',Y_train)
np.save('../ML/data/CS6364HW1digit/test_Kmeans',test)


X_train = np.load('../ML/data/CS6364HW1digit/X_train_Kmeans.npy')
Y_train = np.load('../ML/data/CS6364HW1digit/Y_train_Kmeans.npy')

# split the train set to get validation set
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_validation, Y_validation, test_size=0.5, random_state=1)

# show example
g = plt.imshow(X_train[12][:,:])

# each pixel has 10 centroid, or we can say 10-NN
centroid_list = np.ndarray([28,28,10])
centroid_list.fill(0)

# find digit appearing times in Y_train
digit_train = [0] * 10
for val in Y_train:
    digit_train[val] += 1

def train(X,Y,centroids,digit_memo):
    # count = 0
    row_length = 28
    col_length = 28
    for i in range(len(X)):
        for row in range(row_length):
            for col in range(col_length):
                # count += 1
                centroids[row][col][Y[i]] += X[i][row][col]
    # normalize_centroid
    for row in range(row_length):
        for col in range(col_length):
            for i in range(10):
                centroids[row][col][i] /= digit_memo[i]

    return centroids

centroid_train = train(X_train,Y_train,centroid_list,digit_train)
del centroid_list, X_train, Y_train
np.save('../ML/data/CS6364HW1digit/centroid_train_Kmeans',centroid_train)
np.save('../ML/data/CS6364HW1digit/X_validation_Kmeans',X_validation)
np.save('../ML/data/CS6364HW1digit/Y_validation_Kmeans',Y_validation)
np.save('../ML/data/CS6364HW1digit/X_test_Kmeans',X_test)
np.save('../ML/data/CS6364HW1digit/Y_test_Kmeans',Y_test)

'''
centroid_train = np.load('../ML/data/CS6364HW1digit/centroid_train_Kmeans.npy')
X_validation = np.load('../ML/data/CS6364HW1digit/X_validation_Kmeans.npy')
Y_validation = np.load('../ML/data/CS6364HW1digit/Y_validation_Kmeans.npy')
X_test = np.load('../ML/data/CS6364HW1digit/X_test_Kmeans.npy')
Y_test = np.load('../ML/data/CS6364HW1digit/Y_test_Kmeans.npy')


def validation(X_val,Y_val,centroids):
    row_length = 28
    col_length = 28
    correct_predict = 0
    for i in range(len(X_val)):
        count_panel = [0] * 10
        for row in range(row_length):
            for col in range(col_length):
                cur = X_val[i][row][col]
                for digit in range(10):
                    count_panel[digit] += (centroids[row][col][digit] - cur)**2
        predic_idx = 0
        for digit in range(1,len(count_panel)):
            if count_panel[digit] < count_panel[predic_idx]:
                predic_idx = digit
        #print(predic_idx,Y_val[i])
        if predic_idx == Y_val[i]:
            correct_predict += 1
    print(correct_predict)
    predict_correct_rate = correct_predict / len(Y_val)
    return predict_correct_rate

# The first time thinking is that I should use the distance of the pixel gray level to predict the digit
# But the training accuracy is only 40% which means that the model need improve.
# Then I find the problem is that I take all 0 points not into consideration

# Therefore, after taking 0 into consideration, the accuracy is improved to 60%.
# It is not good yet and I change the evaluation of distance to Euclidean Distance

# With Euclidean distance, the accuracy is improved to 80%
# However, it is still not good enough comparing with NN prediction.



print(validation(X_validation,Y_validation,centroid_train))
print(validation(X_test,Y_test,centroid_train))

