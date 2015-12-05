from numpy import *

def loadData(filename):
    numattr=len(open(filename).readline().strip().split('\t'))-1
    dataMat=[]
    classMat=[]
    fr=open(filename)
    for line in fr.readlines():
        attrs=line.strip().split('\t')
        attrlist=[]
        for i in range(numattr):
            attrlist.append(float(attrs[i]))
        dataMat.append(attrlist)
        classMat.append(float(line(-1)))
    return dataMat,classMat

def standRegress(dataMat,classMat):
    dataMat=mat(dataMat)
    classMat=mat(classMat).transpose()
    xtx=dataMat.transpose()*dataMat
    if linalg.det(xtx)==0.0:
        print("singular !")
        return
    ws=xtx.I*(dataMat.transpose()*classMat)
    return ws