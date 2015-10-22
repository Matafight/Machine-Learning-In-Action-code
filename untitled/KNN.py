# -*- coding: utf-8 -*-
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
def file2matrix(filename):
    fr=open(filename)
    arrayLines=fr.readlines()
    numLines=len(arrayLines)
    attribute=zeros((numLines,3))
    index=0
    classLabels=[]
    for line in arrayLines:
        lines=line.strip() #截取掉所有回车字符
        lists=lines.split('\t')
        attribute[index,:]=lists[0:3]
        classLabels.append(int(lists[-1]))
        index +=1
    return attribute,classLabels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistance=sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    sortedDistanc=distance.argsort()#return the index in the order of the ascending number
    classCount={}
    for i in range(k):
        votelabel=labels[sortedDistanc[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def autoNorm(dataSet):
    minVals=dataSet.min(0)#参数0表示从列种选取最小值
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normData=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normData=dataSet-tile(minVals,(m,1))
    normData=normData/tile(ranges,(m,1))
    return normData,ranges,minVals

def datingClassTest():
    hoRatio=0.1
    datingDataMat,datingLabel = file2matrix('datingTestSet2.txt')
    normData,ranges,minvals=autoNorm(datingDataMat)
    m=normData.shape[0]
    numtest=int(m*hoRatio)
    errcount=0
    for i in range(numtest):
        rclass=classify0(normData[i,:],normData[numtest:m,:],datingLabel[numtest:m],3)
        print ("the classifier came back with %d ,the real answer is %d "%(rclass,datingLabel[i]))
        if (rclass!=datingLabel[i]):
            errcount+=1

    print ("the total error rate is %f" %(errcount/float(numtest)))

if __name__=='__main__':
    '''returnmat,classlabels=file2matrix('datingTestSet2.txt')
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(returnmat[:,1],returnmat[:,2],15.0*array(classlabels),15.0*array(classlabels))
    plt.show()'''
    datingClassTest()