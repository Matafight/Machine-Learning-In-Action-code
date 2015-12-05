from numpy import *

#auto loaddata function
def loadDataSet(filename):
    numattr=len(open(filename).readline().strip().split())-1
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        attrs=[]
        line=line.strip().split()
        for i in range(numattr):
            attrs.append(float(line[i]))
        dataMat.append(attrs)
        #remember to add float
        labelMat.append(float(line[-1]))

    return dataMat,labelMat


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
    #D is a matrix
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
                errArr=mat(ones((m,1)))
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
    m,n=shape(dataArr)
    dataMatrix=mat(dataArr)
    clalabel=mat(classLabels).transpose()
    D=ones((m,1))
    D=D/m

    aggClassEst=zeros((m,1))
    for i in range(numiter):
        #train a model use the initial D
        [bestStump,minError,bestClasEst]=buildStump(dataArr,classLabels,D)
        #bestClasEst is the returned classifier
        #minError is the error rate of current D on the training set

        #calculate the weight of the classifier
        alpha=1.0/2 * log((1-minError)/minError)
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        #update the weights of the samples,clalabel and bestClasEst are matrix
        expn=multiply(-alpha,multiply(clalabel,bestClasEst))
        D=multiply(D,exp(expn))
        #like dot-divide in matlab
        D=D/D.sum()

        #combine the simple classifier to a bigge one
        aggClassEst+=bestClasEst

        #use the new classifier to classify the training data
        aggerror=multiply(sign(aggClassEst)!=clalabel,ones((m,1)))
        aggerror=aggerror.sum()/m
        print("current error is :"+str(aggerror))
        if(aggerror==0):
            break
    return weakClassArr

def adaClassify(newdata,weakClassArr):
    #newdata could be a array
    dataMat=mat(newdata)
    m=shape(dataMat)[0]
    result=mat(zeros((m,1)))
    for classifier in weakClassArr:
         g_x=stumpClassify(dataMat,classifier['dim'],classifier['thresh'],classifier['ineq'])
         result+=multiply(classifier['alpha'],g_x)
    return sign(result)
if __name__=="__main__":
    dataArr,classLabels=loadSimpleData()
    weakClassifier=adaBoostTrainDS(dataArr,classLabels)
    result=adaClassify([[5,5],[0,0]],weakClassifier)
    print(result)
