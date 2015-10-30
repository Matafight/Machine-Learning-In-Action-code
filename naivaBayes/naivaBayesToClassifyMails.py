# -*- coding: utf-8 -*-
__author__ = 'guo'

import re
from naivebayes import *
#prepare the text ,split the text into words
def textParse(bigstring):
    listOfTokens=re.split(r'\w*',bigstring)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]


def spamTest():
    docList=[]
    classList=[]
    fullText=[]

    for i in range(1,26):
        text=textParse(open('email/spam/%d.txt'%i).read())
        docList.append(text)
        classList.append(1)
        fullText.extend(text)

        text2=textParse(open('email/ham/%d.txt'%i).read())

        docList.append(text)
        classList.append(0)
        fullText.extend(text)


    vocabList=createVocabList(docList)
    trainingSet=list(range(50))
    trainMat=[]
    trainClass=[]
    testSet=[]
    #attention
    for i in list(range(10)):
        testindex=int(random.uniform(0,len(trainingSet)))
       # testSet.append(testindex)
        testSet.append(trainingSet[testindex])
        #range object doesn't support item deletion
        del(trainingSet[testindex])

    for i in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList,docList[i]))
        trainClass.append(classList[i])

    p0v,p1v,pab=trainNaiveBayes0(trainMat,trainClass)

    errorcount=0.0
    for i in testSet:
        result=classifyNB(bagOfWords2Vec(vocabList,docList[i]),p0v,p1v,pab)
        if(result != classList[i]):
            errorcount+=1
    print("the errorrate is %f"%(float(errorcount)/len(testSet)))

if __name__=="__main__":
    spamTest()