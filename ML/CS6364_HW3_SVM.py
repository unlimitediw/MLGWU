import sys

import numpy as np
import pandas as pd

sys.path.append("..")
from ML import EntropyGainGenerator
from sklearn import svm
from ML import CS6364_HW3_SVM_Handwork as SH
from ML import KFoldValidation as KF
import matplotlib.pyplot as plt
np.random.seed(1)

'''
1.dataset cleaning and encoding: Abandon samples with missing terms; change discrete (categorical) features to integers; split dataset.
'''

# calculate variance
k = [0.67,.8,.811,.789,.7,.822,.789,.833,.79,.756]
mean = sum(k)/len(k)

variance = 0
for val in k:
    variance += (val - mean)**2
variance /= len(k)
print(variance)

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
print(len(X.T[13]))
# age fnlwgt education
Xnames = ['age','education num','hours','capital gain','capital loss','fnlwgt','sex','married']
Xthis = [33,5,40,0,0,132870,1,0]
# discrete data to integers
# Y
Y_0 = Y == "<=50K"
Y_1 = Y == ">50K"
Y[Y_0] = -1
Y[Y_1] = 1



data = df.values
# character selection
'''
# before calculating information gain
X[2] //= 400
X[10] //= 2000
X[11] //= 2000

Ecal = EntropyGainGenerator.EntropyGainHelper(Y)
charaNameList = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'marital status', 'occupation',
                 'relationship', 'race', 'sex', 'capital gain', 'capital loss', 'hours.per.week', 'native country']
charaEGDic = {}


for i in range(len(X)):
    charaEGDic[charaNameList[i]] = Ecal.calEG(X[i], Y)
sort_key = sorted(charaEGDic, key=lambda x: charaEGDic[x])[::-1]
rankingEG = []
for key in sort_key:
    rankingEG.append([key, format(charaEGDic[key],'.4f')])
for val in rankingEG:
    print(val)
    pass

rankingEG = np.asarray(rankingEG)
plt.figure(figsize= (16,10))
plt.ylabel('Information Gain')
plt.xlabel('Feature')
plt.bar(rankingEG[:,0][::-1],rankingEG[:,1][::-1])

plt.show()

# From the ranking list we know that relationship and marital status are the two features with highest entropy gain with the original data
# However, both of it is not a numeric data, we can not find the linear relationship in it. 
# If these classification data are used as input data set, we need to find a good way to measure it without expand
# Generally speaking, these kind of feature can be divided as several binary feature which is not suitable to svm model
# Thus, now I will select the numeric data with highest entropy gain.
# "fnlwgt" is a little bit special, I will deal with it later. Apart from it, age and capital gain are the features with highest EG
# What about "fnlwgt"? We have the data set with 30162 data and there 4407 different value for "fnlwgt". If there is not mod treatment to this data, it will be meaningless since
# if each data is individually, the EG will very high but meaningless.
# Finally, I will select age and fnlwgt as my first group of feature
'''

from sklearn.model_selection import train_test_split

X = X[[0,4], :]
X = X.T
Y = Y.astype('int')
X, X_show, Y, Y_show = train_test_split(X, Y, test_size=0.97, random_state=15)
pos = np.asarray([X_show[t] for t in range(X_show.shape[0]) if Y_show[t] == 1])
neg = np.asarray([X_show[t] for t in range(X_show.shape[0]) if Y_show[t] == -1])
xaxis = ["<=50k", ">50k"]
yaxis = [len(neg),len(pos)]
#plt.figure(figsize= (16,10))
plt.ylabel('#data')
plt.xlabel('salary')
plt.bar(xaxis,yaxis)
plt.show()
#print(len(neg)/(len(neg) + len(pos)))



def plotData(show=False,thisNeg = neg,thisPos = pos):
    plt.plot(thisNeg[:, 0], thisNeg[:, 1], 'yo', label="<50k")
    plt.plot(thisPos[:, 0], thisPos[:, 1], 'k+', label=">=50k")
    plt.xlabel("age")
    plt.ylabel("education num")
    plt.legend()
    if show: plt.show()

def plotPredictionData(inputX,hypoY):
    a = np.asarray([inputX[t] for t in range(inputX.shape[0]) if hypoY[t] == 1])
    b = np.asarray([inputX[t] for t in range(inputX.shape[0]) if hypoY[t] == -1])
    plt.title("support vectors")

    plotData(thisNeg = b,thisPos = a)
    plt.show()



def plotBoundary(my_svm, xmin, xmax, ymin, ymax, smooth=100, sklearn=False):
    """
    Function to plot the decision boundary for a trained SVM
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the SVM classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    xvals = np.linspace(xmin, xmax, smooth)
    yvals = np.linspace(ymin, ymax, smooth)
    zvals = np.zeros((len(xvals), len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            if not sklearn:
                zvals[i][j] = float(my_svm.predict(np.asarray([xvals[i], yvals[j]])))
            else:
                zvals[i][j] = float(my_svm.predict(np.asarray([[xvals[i], yvals[j]]])))
            # print(zvals[i][j],xvals[i],yvals[j])
    zvals = zvals.transpose()
    plt.contour(xvals, yvals, zvals, [0])
    plt.title("Decision Boundary")
    plt.show()


# validate function
def validate(Xin, YLabel, predictor, sklearn=False):
    m = len(YLabel)
    TruePositive = 0
    FalsePositive = 0
    TrueNegative = 0
    FalseNegative = 0
    predictY = []
    correct = 0
    for i in range(m):
        if not sklearn:
            curPrediction = predictor.predict(np.asarray(Xin[i]))
        else:
            curPrediction = predictor.predict(np.asarray([Xin[i]]))

        if curPrediction == YLabel[i]:
            # print(curPrediction,YLabel[i])
            correct += 1
        # print(curPrediction,YLabel[i])
        # predictY.append(curPrediction)
    # plotPredictionData(Xin,predictY)
    accuracy = correct / m
    return  accuracy


# Data visualization

'''
# external library
for i in range(1,20):
    linear_svm = svm.SVC(C=i * .02,tol = 2, kernel='linear')
    linear_svm.fit(X_show, Y_show)
    plotData()
    plotBoundary(linear_svm, 0, 100, 0, 20, sklearn=True)

for C in range(1,15):
    for sigma in range(1,10):
        sigma = 3
        gamma = np.power(sigma, -3.)
        gaus_svm = svm.SVC(C=C*.3, kernel='rbf', gamma=gamma)
        gaus_svm.fit(X_show[:1000], Y_show[:1000])
        print(C*0.1,validate(X[:1000], Y[:1000], gaus_svm,sklearn= True))

gaus_svm = svm.SVC(C= .1,kernel='rbf',gamma = gamma)
'''

# kernel 1: linear
# SVM_SAMPLE = SH.SVM_HAND(.2, X_show[:100], Y_show[:100], kernel = "linear",tolerance = .5)

print(X.shape,Y.shape)
trainX, validateX,trainY, validateY = train_test_split(X, Y, test_size=0.85, random_state=15)
SVM_SAMPLE = SH.SVM_HAND(1.2, trainX, trainY, tolerance=0.05, kernel='rbf')
localAccuracy = validate(trainX, trainY, SVM_SAMPLE, sklearn=True)
print(localAccuracy)
'''
dataset = KF.KFold(X, Y)
totalAccuracy = 0
for i in range(1,11):
    trainX, trainY, validateX, validateY = dataset.spilt(i)
    C = 0.5
    # kernel 1: linear
    # SVM_SAMPLE = SH.SVM_HAND(C, trainX, trainY, kernel = "linear",tolerance = 0.5)
    # kernel 2: rbf
    # SVM_SAMPLE = SH.SVM_HAND(C, trainX, trainY, tolerance=0.1, kernel='rbf')
    SVM_SAMPLE = svm.SVC(C= .5,tol = .5, kernel='rbf')
    SVM_SAMPLE.fit(trainX,trainY)
    # plotData()
    # plotBoundary(SVM_SAMPLE, 0, 100, 0, 20, smooth=100)

    localAccuracy = validate(validateX,validateY, SVM_SAMPLE,sklearn= True)
    print(localAccuracy)
    #print("C:", i*0.01, "accuracy:", format(localAccuracy, '.3f'))
    totalAccuracy += localAccuracy
totalAccuracy /= 10
print("10-cross-validation-accuracy:",format(totalAccuracy, '.3f'))

'''
'''
C = 0.000001
SVM_SAMPLE = SH.SVM_HAND(C, trainX, trainY, kernel = "polynomial",tolerance = 0.98)
m, accuracy, TruePositive, FalsePositive, TrueNegative,FalseNegative = validate(trainX, trainY, SVM_SAMPLE)
precision = TruePositive/(TruePositive + FalsePositive)
recall = TruePositive/(TruePositive+FalseNegative)
Score = 2 * precision * recall / (precision + recall)
print(TruePositive,FalsePositive, TrueNegative,FalseNegative)
print(accuracy,precision,recall,Score)
#final = np.asarray([np.asarray(totalList), np.asarray(truePositiveList), np.asarray(trueNegativeList),
#                    np.asarray(falsePositiveList), np.asarray(falseNegativeList), np.asarray(TPRList),
#                    np.asarray(TNRList)])
#df = pd.DataFrame(final.T)
#print(df)
#df.index = np.arange(1, len(df) + 1)
#df.to_csv("../ML/data/CS6364HW2Liver/confidenceMetric.csv",
#          index_label=["iteration times", "total test", "truePositive", "trueNegative", "falsePositive",
#                       "falseNegative", "TPR", "TNR"])
plotData()
plotBoundary(SVM_SAMPLE, 0, 100, 0, 20,smooth=100)
'''
# Predict validating and testing

'''
# After checking with all pair of two feature
# 0 2 [age fnlwgt] is the best pair but still not enough so I want to apply PCA in this place
# Here I will show all kinds of numeric feature
#  
'''
