__author__ = 'computer'
import sys
import numpy as np

def sigmoid(x):
    return 1.0 / (1 +  np.exp(-x))

def costDerivation(h, y, X, weight, lmda = 0):
    m, n = np.shape(weight)
    grad = X.transpose().dot(h - y) / m
    #return grad
    pw = weight[1:]
    grad = grad + np.append(0,pw * lmda / m).reshape(m,1)

    return grad
