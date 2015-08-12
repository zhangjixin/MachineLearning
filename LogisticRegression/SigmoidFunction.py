__author__ = 'computer'
import sys
import numpy as np

def sigmoid(x):
    return 1.0 / (1 +  np.exp(-x))

def costFunction(h, y):
    m, n = np.shape(h)
    #print(1-h)
    t1 =  np.multiply(1-y, np.log(1-h))
    t1[np.isnan(t1)] = 0
    t1[np.isinf(t1)] = 0
    #print(t1)
    #print(np.sum(t1))
    #print(y)
    return np.sum(-np.multiply(y, np.log(h)) - t1) / m

def costDerivation(h, y, X, weight, lmda = 0):
    m, n = np.shape(weight)
    grad = X.transpose().dot(h - y) / m
    #return grad
    pw = weight[1:]
    grad = grad + np.append(0,pw * lmda / m).reshape(m,1)

    return grad
