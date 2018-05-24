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

from sklearn import decomposition


def pca(X, k):
    print('mean used in pca:', X.mean(axis=0))
    # Standardize by remove average
    X = X - X.mean(axis=0)

    # Calculate covariance matrix:
    X_cov = np.cov(X.T, ddof=0)
    print('Covariance of the data matrix:')
    print(X_cov)

    # Calculate  eigenvalues and eigenvectors of covariance matrix
    eigenvalues, eigenvectors = eig(X_cov)

    # top k large eigenvectors
    klarge_index = eigenvalues.argsort()[-k:][::-1]
    k_eigenvectors = eigenvectors[klarge_index]
    print('Coordinates of mean after eigenvector transform:',
          np.dot(X.mean(axis=0), k_eigenvectors.T))
    print('eigenvectors:')
    print(k_eigenvectors)
    print('Coordinates of eigenvectors after transform:')
    print(np.dot(k_eigenvectors, k_eigenvectors.T))
    print('Explained variance:', eigenvalues/sum(eigenvalues))

    return np.dot(X, k_eigenvectors.T)


def sk_pca(X, k):
    pca2 = decomposition.PCA(n_components=k, svd_solver='full')

    # X = X - X.mean(axis=0)

    pca2.fit(X)
    print('mean used in pca process:', pca2.mean_)
    print('Coordinates of mean after transform:',
          pca2.transform(np.array([pca2.mean_.tolist()])))
    print('components after mean shift:')
    print(pca2.components_)
    adjust_components = pca2.components_ + pca2.mean_
    print('Coordinates of adjusted components after transform:')
    print(pca2.transform(adjust_components))
    print('Coordinates of eigenvector-like components after transform:')
    print(np.dot(pca2.components_, pca2.components_.T))
    print('Explained variance:', pca2.explained_variance_ratio_)

    X_reduced = pca2.transform(X)

    return X_reduced


def cal_scatter(func, X, k):
    X_pca = func(X, k)
    X_pca = X_pca.tolist()
    xy = zip(*X_pca)
    return xy


if __name__ == '__main__':

    XX = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    print('mean:', XX.mean(axis=0))
    print('std:', XX.std(axis=0))

    iris = load_iris()
    X0 = iris.data
    print('Shape of X0:', X0.shape)
    k0 = 2

    x1y1 = cal_scatter(pca, X0, k0)
    x2y2 = cal_scatter(sk_pca, X0, k0)

    x1, y1 = x1y1
    x2, y2 = x2y2

    print('x1y1:', list(zip(x1, y1)))
    print('x2y2:', list(zip(x2, y2)))

    fig = plt.figure(1)
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(x1, y1, color='blue')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(x2, y2, color='red')
    plt.show()
