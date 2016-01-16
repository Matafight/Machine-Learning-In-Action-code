from numpy import *



def generateDataAuto():
    dataMat=zeros((3,3))
    LcolumMat=[]
    LlabelMat=[]
    IcolumMat=[]
    IlabelMat=[]
    for i in range(3):
        for j in range(3):
            dataMat=zeros((3,3))
            dataMat[i][j]=1
            for k in range(i+1,3):
                dataMat[k][j]=1
                IcolumMat.append(dataMat.copy())
                IlabelMat.append(-1)
                #remember to use the fun of copy!!!
                datatemp=dataMat.copy()
                for p in range(j,3):
                    datatemp[k][p]=1
                    if(p > j):
                        LcolumMat.append(datatemp.copy())
                        LlabelMat.append(1)
    m=shape(LlabelMat)[0]
    n=shape(IlabelMat)[0]
    trainMat=zeros((m+n,9))
    labelMat=zeros((m+n,1))
    for num in range(m+n):
        if num < m:
            trainMat[num]=LcolumMat[num].reshape(9,)
            labelMat[num]=1
        else:
            trainMat[num]=IcolumMat[num-m].reshape(9,)
            labelMat[num]=-1
    return trainMat,labelMat


def trainPer(trainMat,labelMat,numiter=500):
    #w random
    w=zeros((9,1))
    #w[i]is between [-10,10)
    w=20*random.random_sample((9,1))-10
    boolcov=True
    step=1
    lensample=shape(trainMat)[0]
    convergence=False
    for j in range(numiter):
        correctcount=0
        for i in range(lensample):
            fy=sign(dot(trainMat[i],w))
            if(fy!=labelMat[i]):
                #num * matrix?,should use the function multiply
                w=w+step*(labelMat[i]-fy)*trainMat[i].reshape((9,1))
            else:
                correctcount+=1
        if correctcount==lensample:
            break
    if(correctcount!=lensample):
        #print("no convertgence!")
        boolcov=False
    #print("iternum:"+str(j))
    return w,boolcov

def addnoise(trainMat,labelMat):
    #choose a position randomly
    lensamples=shape(trainMat)[0]
    for i in range(lensamples):
        pos=random.randint(0,9,1)
        if(trainMat[i][pos]==1):
            trainMat[i][pos]=0
        else:
             trainMat[i][pos]=1

    return trainMat,labelMat



if __name__=="__main__":
    trainMat,labelMat=generateDataAuto()
    #select 8 samples as test data
    #repeat 10 times
    countcovclean=0
    countcovnoise=0
    for iter in range(1000):
        pos=random.randint(0,shape(trainMat)[0],8)
        forTestMat=trainMat[pos]
        labelTest=labelMat[pos]
        #doesn't delete in place
        newtrainMat=delete(trainMat,pos,0)
        newlabelMat=delete(labelMat,pos,0)

        w,boolconv=trainPer(newtrainMat,newlabelMat)
        if(boolconv):
            countcovclean+=1
        newtrainMat,newlabelMat=addnoise(newtrainMat,newlabelMat)
        #with noise
        w,boolconv2=trainPer(newtrainMat,newlabelMat)
        if(boolconv2):
            countcovnoise+=1
    print(countcovclean)
    print(countcovnoise)


