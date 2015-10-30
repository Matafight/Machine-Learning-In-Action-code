# -*- coding: utf-8 -*-
__author__ = 'guo'

from numpy import *
def loadDataSet():
    fr=open("text.txt")
    arraylines=fr.readlines()
    lenarray=len(arraylines)
    dataSet=[]
    for line in arraylines:
        line=line.strip()
        mylist=line.split(' ')
        dataSet.append(mylist)
    classVec=[0,1,0,1,0,1]
    return dataSet,classVec

def createVocabList(dataSet):
    wordset=set([])
    for item in dataSet:
        wordset=wordset|set(item)
    return list(wordset)  #convert to list ,why?

#word set model
def setOfWords2Vec(vocabList,inputSet):
    inputLen=len(vocabList)
    returnVec=[0]*inputLen  #all of the value of the list is 0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
    return returnVec

#word bag model
def bagOfWords2Vec(vocabList,inputSet):
    returnVec=zeros(len(vocabList))
    for item in inputSet:
        if item in vocabList:
            returnVec[vocabList.index(item)]+=1
    return returnVec

def trainNaiveBayes0(trainMat,trainClass):
    #step1:calculate the number of instance of different classes
    #step2:for specifical class ,calculate the numbers of different attribute
    #step3:calculate the probability of different attribute that appear in different class
    numitems=len(trainMat)
    numwords=len(trainMat[0]) # nearly forget
    numpositive=sum(trainClass)
    #p0num=zeros(numwords)
    #for the purpose of smoothing we usually add one to the initial setup
    p0num=ones(numwords)
    p1num=ones(numwords)
    #p1num=zeros(numwords)

    #initial with 2
    p0demo=2
    p1demo=2
    for i in range(numitems):
        if(trainClass[i]==1):
            p1num+=trainMat[i]
            p1demo+=sum(trainMat[i])
        else:
            p0num+=trainMat[i]
            p0demo+=sum(trainMat[i])
    #to avoid overflow ,we take log with the probability

    p1prob=log(p1num/p1demo)
    p0prob=log(p0num/p0demo)
    ppositive=numpositive/float(numitems)
    return p0prob,p1prob,ppositive


def classifyNB(vec2Classify,prob0,prob1,ppositive):
    #the product sign here means dot product,elementwise
    p1=sum(vec2Classify*prob1)+log(ppositive)
    p0=sum(vec2Classify*prob0)+log(1-ppositive)
    if(p1>p0):
        return 1
    else:
        return 0

def testNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0v,p1v,pab=trainNaiveBayes0(trainMat,listClasses)
    testEntry=['garbage']
    thisDoc=setOfWords2Vec(myVocabList,testEntry)
    print ("classified as :%d"%classifyNB(thisDoc,p0v,p1v,pab))

if __name__=="__main__":
   testNB()
