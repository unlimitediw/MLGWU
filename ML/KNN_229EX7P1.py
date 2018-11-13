import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy.io
import scipy.misc # show matrix as image
from random import sample # random initialization

datafile = 'data/229/ex7data2.mat'
mat = scipy.io.loadmat(datafile)
# print(mat)
X = mat['X']
K = 3 # KNN K = 3
# initial centroids are [3,3],[6,2],[8,5] in ex7.m
initial_centroids = np.array([[3,3],[6,2],[8,5]])
def plotData(myX,mycentroids,myidxs = None):
    colors = ['b','r','y','g','olivedrab','salmon']

    assert myX[0].shape == mycentroids[0][0].shape
    assert mycentroids[-1].shape[0] <= len(colors)

    if myidxs is not None:
        assert myidxs.shape[0] == myidxs.shape[0]
        subX = []
        for x in range(mycentroids[0].shape[0]):
            subX.append(np.array([myX[i] for i in range(myX.shape[0]) if myidxs[i] == x]))
    else:
        subX = [myX]

    fig = plt.figure(figsize= (7,5))
    for x in range(len(subX)):
        newX = subX[x]
        plt.plot(newX[:,0],newX[:,1],'o',color = colors[x],alpha = 0.75, label = 'Data Points: Cluster %d' %x)
    plt.xlabel('x1',fontsize = 14)
    plt.ylabel('x2',fontsize = 14)
    plt.title('Plot of X points', fontsize = 20)
    plt.grid(True)

    tempx, tempy = [],[]
    for mycentroid in mycentroids:
        tempx.append(mycentroid[:,0])
        tempy.append(mycentroid[:,1])
    for x in range(len(tempx[0])):
        plt.plot(tempx,tempy,'rx--',markersize = 8)

    leg = plt.legend(loc = 4,framealpha = 0.5)

#plotData(X,[initial_centroids])
#plt.show()

'''
assert 2 < 1, "1 smaller than 2"
print(1)
'''

def distSquared(point1,point2):
    assert point1.shape == point2.shape
    return np.sum(np.square(point2-point1))

def findClosestCentroids(myX,mycentroids):
    idxs = np.zeros((myX.shape[0],1))
    for x in range(idxs.shape[0]):
        mypoint = myX[x]
        mindist, idx = float('inf'),0
        for i in range(mycentroids.shape[0]):
            dist = distSquared(mycentroids[i],mypoint)
            if dist < mindist:
                mindist = dist
                idx = i
        idxs[x] = idx
    return idxs

# idxs = findClosestCentroids(X,initial_centroids)
# plotData(X,[initial_centroids],idxs)
# plt.show()

def computeCentroids(myX,myidxs):
    subX = []
    for x in range(len(np.unique(myidxs))):
        subX.append(np.array([myX[i] for i in range(myX.shape[0]) if myidxs[i] == x]))
    return np.array([np.mean(thisX,axis =0) for thisX in subX])

def runKMeans(myX, initial_centroids,K,n_iter):
    centroid_history = []
    current_centroid = initial_centroids
    for myiter in range(n_iter):
        centroid_history.append(current_centroid)
        idxs = findClosestCentroids(myX,current_centroid)
        current_centroid = computeCentroids(myX,idxs)
    return idxs,centroid_history

idxs,centroid_history = runKMeans(X,initial_centroids,K =3,n_iter=10)
plotData(X,centroid_history,idxs)
plt.show()

def chooseKRandomCentroids(myX,K):
    rand_indices = sample(range(0,myX.shape[0]),K)
    return np.array([myX[i] for i in rand_indices])

# for x in range(5):
    # idxs, centroid_hostory = runKMeans(X, chooseKRandomCentroids(X,K= 3),K = 3,n_iter= 10)
    # plotData(X,centroid_hostory,idxs)
    # plt.show()

# image data compression
datafile = 'data/229/bird_small.png'
A = imageio.imread(datafile)
print(A.shape)
plt.imshow(A)

A = A / 255
A = A.reshape(-1,3)
myK = 16
idxs, centroid_history = runKMeans(A,chooseKRandomCentroids(A,myK),myK,n_iter=10)
idxs = findClosestCentroids(A,centroid_history[-1])
final_centroid = centroid_history[-1]

final_image = np.zeros((idxs.shape[0],3))
for x in range(final_image.shape[0]):
    final_image[x] = final_centroid[int(idxs[x])]
plt.figure()
plt.imshow(A.reshape(128,128,3))
plt.show()
plt.figure()
plt.imshow(final_image.reshape(128,128,3))
plt.show()
