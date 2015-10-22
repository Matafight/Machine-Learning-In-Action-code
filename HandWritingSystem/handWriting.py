# -*- coding: utf-8 -*-
from numpy import *
from os import listdir
import operator
def img2vector(file):
    retVec=zeros((1,1024))
    fr=open(file)
    for i in range(32):
        line=fr.readline()
        for j in range(32):
            retVec[0,32*i+j]=int(line[j])
    return retVec
def classify0(inX,trainMat,labels,k):
    m=trainMat.shape[0]
    tarMat=tile(inX,(m,1))
    diffMat=trainMat-tarMat
    sqMat=diffMat**2
    sumMat=sqMat.sum(axis=1) #add the rows
    distance=sumMat**0.5
    sortedDistance=distance.argsort() # return the index in ascending order
    classCount={}
    for i in range(k):                 #choose the first k
        votelabel=labels[sortedDistance[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1
    sortedList=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedList[0][0]

#auto norm 归一化
def autoNorm(dataSet):
    minval=dataSet.min(0)
    maxval=dataSet.max(0)
    ranges=maxval-minval
    normdata=zeros(shape(dataSet))
    normdata=dataSet-tile(minval,(dataSet.shape[0],1))
    normdata=normdata/tile(ranges,(dataSet.shape[0],1))
    return normdata,ranges,minval

def handWritingClassTest():
    hwlabels=[]
    trainingFilelist=listdir('trainingDigits')
    m=len(trainingFilelist)
    trainMat=zeros((m,1024))
    for i in range(m):
        filenamestr=trainingFilelist[i]
        fname=filenamestr.split('.')[0]
        classlabel=int(fname.split('_')[0])
        hwlabels.append(classlabel)
        trainMat[i,:]=img2vector('trainingDigits/%s'%filenamestr)
    testFilelist=listdir('testDigits')
    errorcount=0.0
    testlen=len(testFilelist)
    for i in range(testlen):
        testfilename=testFilelist[i]
        testfname=testfilename.split('.')[0]
        testlabel=int(testfname.split('_')[0])
        retlabel=classify0(img2vector('testDigits/%s'%testfilename),trainMat,hwlabels,3)
        print('the classifier came back with %d the real answer is %d'%(testlabel,retlabel))
        if(retlabel!=testlabel):
            errorcount+=1
    print('the total number of errors is %f'%errorcount)
    print('the total error rate is %f'%(errorcount/testlen))


if __name__=='__main__':
    handWritingClassTest()
