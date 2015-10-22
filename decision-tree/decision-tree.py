# -*- coding: utf-8 -*-
__author__ = 'guo'

from math import log
import operator
def createDataSet():
    dataset=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataset,labels

def calculateShannonEnt(dataSet):
    numitems=len(dataSet)
    labelCount={}
    entory=0.0
    for item in dataSet:
        clabel=item[-1]
        labelCount[clabel]=labelCount.get(clabel,0)+1
    #what is the difference for key ,id in enumrate ???????
    for key in labelCount:
        prob=float(labelCount[key])/numitems
        entory-=prob*log(prob,2)
    return entory

def splitData(dataSet,pos,val):
    retMat=[]
    retitem=[]
    for data in dataSet:
        if(data[pos]==val):
            retitem=data[0:pos]
            retitem.extend(data[pos+1:])
            retMat.append(retitem)
    return retMat

#calculate the most shrank of shannon
#type(dataSet) : list
def chooseBestFeatureToSplit(dataSet):
    numfeature=len(dataSet[0])-1
    basentroy=calculateShannonEnt(dataSet)
    newentroy=0.0
    gainentroy=0.0
    maxgain=0.0
    targetfea=-1
    lendata=len(dataSet)
    for i in range(numfeature):
        feaval=[example[i] for example in dataSet]
        uniqueVal=set(feaval)
        newentroy=0.0
        for val in uniqueVal:
            retMat=splitData(dataSet,i,val)
            prob=len(retMat)/float(lendata)
            newentroy+=prob*calculateShannonEnt(retMat)
        gainentroy=basentroy-newentroy
        if(gainentroy > maxgain):
            maxgain=gainentroy
            targetfea=i
    return targetfea

#return the majority class
def majorityCount(classList):
    classCount={}
    for vote in classList:
        classCount[vote]=classCount.get(vote,0)+1
    sortClassList=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortClassList[0][0]

#the parameter of labels is the list of the name of the attributes
def createTree(dataSet,labels):
    labellist=[example[-1] for example in dataSet]
    if(labellist.count(labellist[0])==len(labellist)):
        return labellist[0]
    if(len(dataSet[0])==1):
        return majorityCount(labellist)

    bestfeat=chooseBestFeatureToSplit(dataSet)
    #bestfeat is the index bestfeatlabel is the name
    bestfeatlabel=labels[bestfeat]
    myTree={bestfeatlabel:{}}
    featval=[example[bestfeat] for example in dataSet]
    uniqueval=set(featval)
    for itemval in uniqueval:
        sublabel=labels[:]
        myTree[bestfeatlabel][itemval]=createTree(splitData(dataSet,bestfeat,itemval),sublabel)
    return myTree


if __name__=='__main__':
    dataset,label=createDataSet()
    #entroy=calculateShannonEnt(dataset)
    #print('entroy:%f'%entroy)
    #ret=chooseBestFeatureToSplit(dataset)
    dic=createTree(dataset,label)
    print (dic)
    #print("best feature:%d"%ret)