import os
import sys
import numpy as np
import _pickle
import gzip

def load_data():
    fr = gzip.open('mnist.pkl.gz', 'rb')
    train_data, validation_data, test_data = _pickle.load(fr, encoding='latin1')
    fr.close()

    train_img            =    [np.reshape(pixels, (784, 1)) for pixels in train_data[0]]
    train_label          =    [vectorize(label) for label in train_data[1]]
    train_data           =    [[x, y] for x, y in zip(train_img, train_label)]

    validation_img       =    [np.reshape(pixels, (784, 1)) for pixels in validation_data[0]]
    validation_label     =    [vectorize(label) for label in validation_data[1]]
    validation_data      =    [[x,y] for x, y in zip(validation_img, validation_label)]

    test_img             =    [np.reshape(pixels, (784, 1)) for pixels in test_data[0]]
    #test_label           =    [vectorize(label) for label in test_data[1]]
    test_label           =    [label for label in test_data[1]]
    test_data            =    [[x, y] for x, y in zip(test_img, test_label)]

    return (train_data, validation_data, test_data)

def vectorize(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


