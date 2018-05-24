"""
@Project   : DuReader
@Module    : pca.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/24/18 1:16 PM
@Desc      : 
"""

'''
因为用了numpy包，算法的实现非常简单，如果纯粹用C写的话要复杂的多。

首先对向量X进行去中心化
接下来计算向量X的协方差矩阵，自由度可以选择0或者1
然后计算协方差矩阵的特征值和特征向量
选取最大的k个特征值及其特征向量
用X与特征向量相乘
'''

"""
Created on Mon Sep  4 19:58:18 2017

@author: zimuliu
http://blog.jobbole.com/109015/
"""

from sklearn.datasets import load_iris
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt


def pca(X, k):
    # Standardize by remove average
    X = X - X.mean(axis=0)

    # Calculate covariance matrix:
    X_cov = np.cov(X.T, ddof=0)

    # Calculate  eigenvalues and eigenvectors of covariance matrix
    eigenvalues, eigenvectors = eig(X_cov)

    # top k large eigenvectors
    klarge_index = eigenvalues.argsort()[-k:][::-1]
    k_eigenvectors = eigenvectors[klarge_index]

    return np.dot(X, k_eigenvectors.T)


iris = load_iris()
X = iris.data
k = 2

X_pca = pca(X, k)
X_pca = X_pca.tolist()
x, y = zip(*X_pca)

# print(X_pca)

from sklearn import decomposition

pca2 = decomposition.PCA(n_components=k)
X_reduced = pca2.fit_transform(X)
X_pca2 = X_reduced.tolist()
x2, y2 = zip(*X_pca)

fig = plt.figure(1)
ax = fig.add_subplot(1, 2, 1)
ax.scatter(x, y)
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(x2, y2)
plt.show()
