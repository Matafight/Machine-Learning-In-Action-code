# -*- coding: utf-8 -*-
__author__ = 'guo'


from numpy import *
def loadData():
    trainMat=[]
    labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        item=line.strip().split()
        trainMat.append([1.0,float(item[0]),float(item[1])])
        labelMat.append(int(item[2]))
    return trainMat,labelMat

def sigmoid(x):
    return 1.0/(1+exp(-1*x))

def gradAscend(trainMat,classLabel):
    trainMat=mat(trainMat)
    #now the shape of classLabel is 100 *１，a array vector
    classLabel=mat(classLabel).transpose()
    lenitem,lenattr=shape(trainMat)
    #type(trainMat) is matrix now，and to find its row and array ,we need to call the function shape
    #the return type of ones is array,array * array is a element-wise calculation,mat*array is the matrix multi

    weights=ones((lenattr,1))
    ylabel=zeros((lenitem,1))
    alpha=0.001
    maxCal=500
    for i in range(maxCal):
        #numpy 中的矩阵相乘,the return type is matrix
        ytemp=trainMat*weights
        ylabel=sigmoid(ytemp)
        error=classLabel-ylabel
        weights=weights+alpha*trainMat.transpose()*error
    return weights

#draw the decision boundary
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadData()
    dataMat=mat(dataMat)
    labelMat=mat(labelMat).transpose()
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    m,n=shape(dataMat)
    for i in range(m):
        if(labelMat[i]==1):
            xcord1.append(trainMat[i][1])
            ycord1.append(trainMat[i][2])
        else:
            xcord2.append(trainMat[i][1])
            ycord2.append(trainMat[i][2])
    #what's the meaning behind this code?
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    #like linspace in matlab
    x=array(arange(-3.0,3.0,0.1))
    y=((-weights[0]-weights[1]*x)/weights[2]).transpose()
    ax.plot(x,y)
    plt.show()

if(__name__=="__main__"):
    trainMat,labelMat=loadData()
    weights=gradAscend(trainMat,labelMat)
    print(weights)
    plotBestFit(weights)
