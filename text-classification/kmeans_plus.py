"""
@Project   : DuReader
@Module    : kmeans_plus.py
@Author    : Deco [deco@cubee.com]
@Created   : 6/20/18 3:50 PM
@Desc      : https://zhuanlan.zhihu.com/p/31245139
"""
"""
Created on Fri May  5 21:13:05 2017

@author: liuzimu
"""
"""
Article to introduce Kmeans++ Algorithm
http://www.csdn.net/article/2012-07-03/2807073-k-means
"""
import pprint
import numpy as np
"""
Part 1
set initial k-means centers
"""

"""
1.1 aux func
calculate the disctance from each point to kms_center
"""

def euc_distance(X, n_row, kms_center):
    D = ((X - kms_center) ** 2).sum(axis = 1) ** 0.5
    return D

"""
1.2 aux func
randomly choose a point from input data as an initial kmeans center
"""
def rnd_center(X, n_row, last_center):
    # calculate euc distance of dataset
    D = euc_distance(X,n_row,last_center)
    D_sqr = D * D
    D_sqr_cum = D_sqr.cumsum()
    D_sqr_sum = D_sqr.sum()
    # randomly choose a number, whose value in [D[0,],D_sum]
    rnd = np.random.rand() * (D_sqr_sum - D_sqr[0,]) + D_sqr[0,]
    # calculate the index by rnd and D_cum
    ix_rnd = np.argwhere(D_sqr_cum <= rnd)
    return ix_rnd[-1,0]
"""
1.3 main func
calculate initial k-means centers
"""
def init_center(X,k,n_row,n_column):
    kms_center = np.empty((k,n_column),'float')
    # randomly choose a point from input data as 1st initial kmeans center
    rnd = np.empty(k,'int')
    r = np.random.choice(range(n_row))
    rnd[0] = r
    kms_center[0,] = X[r]
    # randomly choose 2nd to k th centers
    i = 1
    while i < k:
        r  = rnd_center(X,n_row,kms_center[i-1,])
        # in case of duplicated random number
        if r in rnd:
            pass
        else:
            rnd[i] = r
            kms_center[i,] = X[r,]
            i += 1
    return kms_center
"""
Part 2
calculate final k-means centers
"""
"""
2.1 aux func
calculate new kmeans center and group
The method of kmeans++
"""
def shift_center(X,k,kms_center,n_row):
    # distance from each kmeans center
    D = np.empty((n_row,k),'float')
    # calculate d_array
    for i in range(k):
        D[:,i] = euc_distance(X,n_row,kms_center[i,])
    # nearest kmeans center of each row
    grp = D.argmin(axis = 1)
    # sum of x of each group
    x_sum_grp = np.zeros_like(kms_center,'float')
    # count of x of each group
    x_cnt_grp = np.zeros((k,1),'int')
    for i in range(n_row):
        x_sum_grp[grp[i],] += X[i,]
        x_cnt_grp[grp[i],] += 1
    # calculate new kmeans center
    kms_center_new = x_sum_grp/x_cnt_grp
    #calculate new kmeans center for emtpy group
    empty_grp = np.argwhere(x_cnt_grp == 0)[:,0]
    for i in empty_grp:
        d_tot = D.sum(axis = 1)
        kms_center_new[i,] = X[d_tot.argmax(),]
        for j in range(k):
            if np.isnan(kms_center_new[j,0]):
                D[:,j] = euc_distance(X,n_row,kms_center_new[j,])
            else:
                D[:,j] = 0
    return kms_center_new,grp
"""
2.2 main func
calculate final kmeans center and group
"""
def kMeans(X,k,MaxIter):
    n_row,n_column = X.shape
    kms_center = init_center(X,k,n_row,n_column)
    for i in range(MaxIter):
        kms_center_new,grp = shift_center(X,k,kms_center,n_row)
        if np.all(kms_center == kms_center_new):
            break
        else:
            kms_center = kms_center_new
    return kms_center, grp
"""
Part 3 compare with sklearn kmeans
"""
"""
3.1 result check
"""
# import dataset iris
from sklearn.datasets import load_iris
# my kMeans
iris = load_iris()
kMeans(iris.data, 2, 100)
# sklearn
from sklearn.cluster import KMeans
def sklearn_kMeans():
    model = KMeans(n_clusters=2, random_state=9)
    model.fit(iris.data)
    return model.cluster_centers_


import timeit


def wrapper(func, *args, **kwargs):
    """
    一个装饰器，用来把有参数的函数转换为没有参数的函数
    :param func: func
    :param args: list
    :param kwargs: dict
    :return: func
    """
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


if __name__ == '__main__':
    print('Result of sklearn_kMeans():')
    print(sklearn_kMeans())
    print()

    print('Result of kMeans():')
    pprint.pprint(kMeans(iris.data, 2, 100))
    print()

    iterations = 100
    wrapped_sklearn_kMeans = wrapper(sklearn_kMeans)
    wrapped_kMeans = wrapper(kMeans, iris.data, 2, 100)
    print('sklearn_kMeans time cost: {} s'.format(
        timeit.timeit(wrapped_sklearn_kMeans, number=iterations) / iterations))
    print('kMeans++ time cost: {} s'.format(
        timeit.timeit(wrapped_kMeans, number=iterations) / iterations))
