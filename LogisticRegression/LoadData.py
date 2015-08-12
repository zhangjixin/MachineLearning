__author__ = 'computer'
import sys

def loadData(filepath):
    dataDic = []
    labelDic = []
    fr = open(filepath, "r")
    for line in fr.readlines():
        ps = line.strip().split('\t')
        n = len(ps)
        val = [float(1.0)]
        val += [float(ps[i]) for i in range(n-1)]
        dataDic.append(val)
        labelDic.append(int(float(ps[n-1])))
    fr.close()
    return dataDic, labelDic