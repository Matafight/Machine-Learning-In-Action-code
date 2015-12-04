from numpy import *


def loadSimpleData():
    dataMat=mat([[1.0,2.1],[2.0,1.1],[1.3,1],[1,1],[2,1]])
    classLabel=[1,1,-1,-1,1]
    return dataMat,classLabel

#create decision stump
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr)
    labelMat=mat(classLabels).transpose()
    [m,n]=shape(dataMatrix)
    numSteps=10
    bestStump={}
    bestClasEst=mat(zeros((m,1)))
    minError=inf
    for i in range(n):
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequ in ['lt','gt']:
                threshVal=rangeMin+float(j)*stepSize
                perdictedVal=stumpClassify(dataMatrix,i,threshVal,inequ)
                errArr=ones((m,1))
                errArr[perdictedVal==labelMat]=0
                weightErr=D.transpose()*errArr
                if weightErr< minError:
                    minError=weightErr
                    bestClasEst=perdictedVal.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequ
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numiter=40):
    weakClassArr=[]
