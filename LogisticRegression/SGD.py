__author__ = 'computer'
import sys
import numpy as np
import math
import random
from SigmoidFunction import sigmoid, costDerivation

def gd(dataDic, labelDic, alpha = 0.01, epochs = 150):
    trainData = np.mat(dataDic)
    labelData = np.mat(labelDic).transpose()
    m, n = np.shape(trainData)
    weights = np.ones((n,1))

    for k in range(epochs):
        h = sigmoid(trainData * weights)
        #error = (labelData - h)
        error = costDerivation(labelData, h, trainData, weights)
        weights = weights + alpha  *  error
    return weights

def sgd(dataDic, labelDic, alpha = 0.01, epochs = 150):
    trainData = np.mat(dataDic)
    labelData = np.mat(labelDic).transpose()
    m, n = np.shape(trainData)
    weights = np.ones((n,1))

    for k in range(epochs):
        dataIdx = list(range(m))
        for i in range(m):
            alpha = 4 / (1+k+i) + 0.01
            randIdx = int(random.uniform(0, len(dataIdx)))
            data = trainData[randIdx]
            A = data * weights
            h = sigmoid(np.sum(A))
            error = np.sum(labelData[randIdx] - h)
            data = data.transpose()
            weights = weights + alpha * error * data / m
            del(dataIdx[randIdx])
    return weights
