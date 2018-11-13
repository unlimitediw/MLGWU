import math

class EntropyGainHelper(object):
    def __init__(self,Y):
        self.Entropy = self.calEntropy(Y)

    def calEntropy(self, Y):
        m = len(Y)
        typeDic = {}
        for elem in Y:
            if elem not in typeDic:
                typeDic[elem] = 1
            else:
                typeDic[elem] += 1
        res = 0
        for key in typeDic.keys():
            res -= typeDic[key] / m * math.log2(typeDic[key] / m)
        return res

    # attention: input X should be transformed to X.T previously
    # then C = X[i]
    def calEG(self, C, Y):
        charTypeDic = {}
        m = len(Y)
        res = self.Entropy
        for i in range(m):
            if C[i] not in charTypeDic:
                charTypeDic[C[i]] = [Y[i]]
            else:
                charTypeDic[C[i]].append(Y[i])
        for key in charTypeDic.keys():
            res -= len(charTypeDic[key])/m * self.calEntropy(charTypeDic[key])
        return res
