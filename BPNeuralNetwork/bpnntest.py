__author__ = 'computer'
import os
import sys
import bpnnloaddata
import bpneuralnetwork

def main():
    train_data, validation_data, test_data = bpnnloaddata.load_data()
    bpnn = bpneuralnetwork.BPNeuralNetwork([784, 30, 10])
    bpnn.TrainModel(train_data, 30, 10, 0.1, lmbda=5.0)
    cnt = 0
    for x, y in test_data:
        label = bpnn.RunBPNeuralNetwork(x)
        ret = 0
        num = label[0]
        for i in range(0, len(label)):
            if (label[i] > num):
                num = label[i]
                ret = i
        print("{0} {1}".format(ret, y))
        cnt += 1
        if (cnt == 20):
            break

if (__name__ == '__main__'):
    main()