import matplotlib.pyplot as plt


Y1 = [0.86, 0.847,0.833, 0.81,0.801]
Y2 = [0.75, 0.77, 0.785,0.79,0.801]
X = ['1k','2k','3k','4k','5k']

plt.plot(X,Y1)
plt.plot(X,Y1,'bx',label = "training accuracy")
plt.plot(X,Y2,'g')
plt.plot(X,Y2,'gx',label = "cross-validation accuracy")
plt.legend()
plt.title("Learning curve of svm with rbf kernel")
plt.xlabel("Number of training sample")
plt.ylabel("Accuracy")
plt.ylim([0.6,1.0])
plt.show()