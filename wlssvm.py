# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 09:30:24 2018

@author: lj
"""
from numpy import *

def loadDataSet(filename):
    '''导入数据
    input: filename:文件名
    '''
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append(float(lineArr[0]))
        labelMat.append(float(lineArr[1]))
    return mat(dataMat).T,mat(labelMat).T
           

def kernelTrans(X,A,kTup):
    '''数据集中每一个数据向量与A的核函数值
    input: X--特征数据集
           A--输入向量
           kTup--核函数参量定义
    output: K--数据集中每一个数据向量与A的核函数值组成的矩阵
    '''
    X = mat(X)
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K/(-1 * kTup[1] ** 2))
    else: raise NameError('Houston We Have a Problem ,That Kernel is not recognized')
    return K
    
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.K = mat(zeros((self.m,self.m)))  #特征数据集合中向量两两核函数值组成的矩阵，[i,j]表示第i个向量与第j个向量的核函数值
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
            

def leastSquares(dataMatIn,classLabels,C,kTup):
    '''最小二乘法求解alpha序列
    input:dataMatIn:特征数据集
          classLabels:分类标签集
          C:参数，（松弛变量，允许有些数据点可以处于分隔面的错误一侧)
          kTup: 核函数类型和参数选择 
    output:b--w.T*x+b=y中的b
           alphas:alphas序列      
    '''
    ##1.参数设置
    oS = optStruct(dataMatIn,classLabels,C,kTup)
    unit = mat(ones((oS.m,1)))  #[1,1,...,1].T
    I = eye(oS.m)
    zero = mat(zeros((1,1)))
    upmat = hstack((zero,unit.T))
    downmat = hstack((unit,oS.K + I/float(C)))
    ##2.方程求解
    completemat = vstack((upmat,downmat))  #lssvm中求解方程的左边矩阵
    rightmat = vstack((zero,oS.labelMat))    # lssvm中求解方程的右边矩阵
    b_alpha = completemat.I * rightmat
    ##3.导出偏置b和Lagrange乘子序列
    oS.b = b_alpha[0,0]
    for i in range(oS.m):
        oS.alphas[i,0] = b_alpha[i+1,0]
    e = oS.alphas/C
    return oS.alphas,oS.b,e

def weights(e):
    '''计算权重序列
    input:e(mat):LSSVM误差矩阵
    output:v(mat):权重矩阵
    '''
    ##1.参数设置
    c1 = 2.5
    c2 = 3
    m = shape(e)[0]
    v = mat(zeros((m,1)))
    v1 = eye(m)
    q1 = int(m/4.0)
    q3 = int((m*3.0)/4.0)
    e1 = []
    shang = mat(zeros((m,1)))
    ##2.误差序列从小到大排列
    for i in range(m):
        e1.append(e[i,0])
    e1.sort()
    ##3.计算误差序列第三四分位与第一四分位的差
    IQR = e1[q3] - e1[q1]
    ##4.计算s的值
    s = IQR/(2 * 0.6745)
    ##5.计算每一个误差对应的权重
    for j in range(m):
        shang[j,0] = abs(e[j,0]/s)
    for x in range(m):
        if shang[x,0] <= c1:
            v[x,0] = 1.0
        if shang[x,0] > c1 and shang[x,0] <= c2:
            v[x,0] = (c2 - shang[x,0])/(c2 - c1)
        if shang[x,0] > c2:
            v[x,0] = 0.0001
        v1[x,x] = 1/float(v[x,0])
    return v1

def weightsleastSquares(dataMatIn,classLabels,C,kTup,v1):
    '''最小二乘法求解alpha序列
    input:dataMatIn:特征数据集
          classLabels:分类标签集
          C:参数，（松弛变量，允许有些数据点可以处于分隔面的错误一侧)
          kTup: 核函数类型和参数选择 
    output:b--w.T*x+b=y中的b
           alphas:alphas序列      
    '''
    ##1.参数设置
    oS = optStruct(dataMatIn,classLabels,C,kTup)
    unit = mat(ones((oS.m,1)))  #[1,1,...,1].T
    #I = eye(oS.m)
    gamma = kTup[1]
    zero = mat(zeros((1,1)))
    upmat = hstack((zero,unit.T))
    downmat = hstack((unit,oS.K + v1/float(C)))
    ##2.方程求解
    completemat = vstack((upmat,downmat))  #lssvm中求解方程的左边矩阵
    rightmat = vstack((zero,oS.labelMat))    # lssvm中求解方程的右边矩阵
    b_alpha = completemat.I * rightmat
    ##3.导出偏置b和Lagrange乘子序列
    oS.b = b_alpha[0,0]
    for i in range(oS.m):
        oS.alphas[i,0] = b_alpha[i+1,0]
    e = oS.alphas/C
    return oS.alphas,oS.b


def predict(alphas,b,dataMat):
    '''预测结果
    input:alphas(mat):WLSSVM模型的Lagrange乘子序列
          b(float):WLSSVM模型回归方程的偏置
          dataMat(mat):测试样本集
    output:predict_result(mat):测试结果
    '''
    m,n = shape(dataMat)
    predict_result = mat(zeros((m,1)))
    for i in range(m):
        Kx = kernelTrans(dataMat,dataMat[i,:],kTup)   #可以对alphas进行稀疏处理找到更准确的值        
        predict_result[i,0] =  Kx.T * alphas + b   
    return predict_result

def predict_average_error(predict_result,label):
    '''计算平均预测误差
    input:predict_result(mat):预测结果
          label(mat):实际结果
    output:average_error(float):平均误差
    '''
    m,n = shape(predict_result)
    error = 0.0
    for i in range(m):
        error += abs(predict_result[i,0] - label[i,0])
    average_error = error / m
    return average_error
    


if __name__ == '__main__':
    ##1.数据导入
    print('--------------------Load Data------------------------')
    dataMat,labelMat = loadDataSet('sine.txt')
    ##2.参数设置
    print('--------------------Parameter Setup------------------')
    C = 0.6
    k1 = 0.3
    kernel = 'rbf'
    kTup = (kernel,k1)
    ##3.求解LSSVM模型
    print('-------------------Save LSSVM Model-----------------')
    alphas,b,e = leastSquares(dataMat,labelMat,C,kTup)
    ##4.计算误差权重
    print('----------------Calculate Error Weights-------------')
    v1 = weights(e)
    ##5.求解WLSSVM模型
    print('------------------Save WLSSVM Model--------------- -')
    alphas1,b1 = weightsleastSquares(dataMat,labelMat,C,kTup,v1)
    ##6.预测结果
    print('------------------Predict Result------------------ -')
    predict_result = predict(alphas1,b1,dataMat)
    ##7.平均误差
    print('-------------------Average Error------------------ -')
    average_error = predict_average_error(predict_result,labelMat)
    





