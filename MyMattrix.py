# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:37:59 2019
@author: Ai酱
"""
class Mattrix:
    val = []
    shape=(0,)
    def __init__(self,val):
        if type(val)==list:
            self.val = val
        if type(val)==Mattrix:
            self.val = val.val
        self.update_shape()
    def __str__(self):
        return str(self.val)
    def __repr__(self):
        return self.__str__()

    def update_shape(self):
        '''
        更新矩阵的形状属性值
        '''
        self.shape=(len(self.val),)
        if len(self.val)>0 and type(self.val[0])==list:
            self.shape += (len(self.val[0]),)

        
    def size(self):
        return self.shape
    
    def get(self,r,c):
        '''
        r 行(row)
        c 列(col)
        return val[r][c]
        '''
        return self.val[r][c]
    def set(self,r,c,elem_value):
        '''
        设置第r行c列的元素值
        '''
        self.val[r-1][c-1]=elem_value
        self.update_shape()
    def row(self,r):
        if len(self.shape)==1:
            return self.val
        return self.val[r-1]
    
    def col(self,c):
        if len(self.shape)==1:
            return self.val
        else:
            result = [self.val[i][c-1] for i in range(self.shape[0])]

            return result
    def create_mattrix(shape):
        result = None
        for s in shape:
            result = [result]*s
        return Mattrix(result)
    def transpose(self):
        '''
        转置
        '''
        if len(self.size())==1:
            return Mattrix(self.val)
        else:
            # 目前的每列当做新矩阵的行
            result = Mattrix([self.col(i+1) for i in range(self.shape[1])])
            return result
    def operator(self,vector_operator,mat_operator,other,result):
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                    if type(other)==int or type(other)==float:
                        result.val[i][j] = vector_operator(result.val[i][j] , other)
                    else:
                        result.val[i][j] = mat_operator(i,j)
        return result
    
    def add(self,other):
        def matadd(i,j):
            return self.val[i][j]+other.val[i][j]
        pass
        return self.operator((lambda a,b: a+b),matadd,other,Mattrix(self.val))
    
    def dot(vector1,vector2):
        '''
        两个向量相乘
        '''
        v1 = vector1
        v2 = vector2
        if type(vector1)==Mattrix:
            v1 = vector1.val
        if type(vector2)==Mattrix:
            v2 = vector2.val
        if type(v1)==int or type(v1)==float:
            return v1*v2

        result = 0
        for i in range(len(v1)):
            result += v1[i]*v2[i]
        return result
        
    def mul(self,other):
        print(self.val,other.val)
        def matmul(i,j):
            return Mattrix.dot(self.row(i),other.col(j))
        return self.operator((lambda a,b: a*b),matmul,other,Mattrix.create_mattrix((self.shape[0],other.shape[-1])))
    
    def __add__(self,other):
        return self.add(other)
    def __mul__(self,other):
        return self.mul(other)
    def __sub__(self,other):
        return self.add(-1*other)

mat = Mattrix([[4,5,6],[2,3,4]])
x = mat.transpose()
print(mat*x)

