from numpy import *

trainMat=ones((5,6))
w=array([1,2,3,4,5,6])
update=ones((6,1))
ret=trainMat[0]*w


print(ret)
print(type(trainMat[0]))
ret2=dot(trainMat[0],w)
print(ret2)
ret3=3*trainMat[0]
print(ret3)

update=update+2*trainMat[0].reshape(6,1)
print(update)
dataMat=zeros((3,3))
print(dataMat[0][0])

a=random.randint(0,9,10)
print(a)

n=1
c=not(n)
print(c)