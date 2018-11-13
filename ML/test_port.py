test = []
import random
import numpy as np
np.random.seed(1)
def quickSelect(originList,k):
    if not originList:
        return None
    if k < 0 or k > len(originList) - 1:
        raise IndexError()
    return partition(originList,0,len(originList)- 1,k)

def partition(originList,left,right,k):
    if left == right:
        return originList[left]
    pivot = np.random.randint(left,right)
    #pivot = right
    memo = originList[pivot]
    # swap right and pivot
    originList[right], originList[pivot] = originList[pivot], originList[right]
    step = left
    for i in range(left+1,right+1):
        if originList[i] < memo:
            step += 1
            originList[step],originList[i] = originList[i],originList[step]
    originList[step],originList[right] = originList[right],originList[step]
    if k == step:
        return originList[k]
    elif k < step:
        return partition(originList,left,right - 1,k)
    else:
        return partition(originList,left + 1,right,k)


for i in range(1000001):
    test.append(np.random.randint(1,99999))

a = np.asarray(test)
print(np.median(a))


test.sort()
print(test[500000])



# hand writing quickSelect which should be O(n) is much slower than build-in sorted() which should be O(nlogn) due to
# the build-in one is c process actually and quickSelect need recursion which is slow
# Finally I decide to use numpy.median to find the median which is also O(n)
# when the data come to 10^6 level, np.mean takes 14ms and sort takes 608ms.
# 10^6 * log10^6. the deference should be only 6 times but take memory access and different build-in environment into
# consideration, np.median is much faster than .sort().