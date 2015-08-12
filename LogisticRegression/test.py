__author__ = 'computer'
import sys
import numpy as np
import SGD
import LoadData

dataDic, labelDic = LoadData.loadData("horseColicTraining.txt")
w = SGD.gd(dataDic, labelDic, alpha = 0.001, epochs = 1000)
#w = SGD.sgd(dataDic, labelDic, alpha = 0.001, epochs = 500)
dataDic, labelDic = LoadData.loadData("horseColicTest.txt")
h = np.mat(dataDic).dot(w)
cnt = 0
for i in range(len(labelDic)):
    if h[i] >= 0.5 and labelDic[i] >= 0.5:
        cnt += 1
    elif h[i] < 0.5 and labelDic[i] <= 0.5 :
        cnt += 1
print(cnt / len(labelDic))