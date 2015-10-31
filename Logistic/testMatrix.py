# -*- coding: utf-8 -*-
__author__ = 'guo'

from numpy import *
def test():
    mat1=array([[1,2],[3,4]])
    array2=mat(ones((2,2)))
    result=mat1*array2
    return result

if(__name__=="__main__"):
    print(test())
    print(type(test()))