# -*- coding: utf-8 -*-
__author__ = 'guo'

import re
from naivebayes import *
#prepare the text ,split the text into words
def textParse(bigstring):
    listOfTokens=re.split(r'\w*',bigstring)
    return [tok.lower() for tok in listOfTokens]


def spamTest():
    docList=[]
    classList=[]
    fullText=[]

    for i in range(26):
        text=textParse(open('email/spam/%d.txt'%i).read())
        docList.append(text)
        classList.append(1)
        fullText.extend(text)

        text2=textParse(open('email/ham/%d.txt'%i).read())
        docList.append(text)
        classList.append(0)
        fullText.extend(text)

    vocabList=createVocabList(docList)
    trainingSet=range(50)
