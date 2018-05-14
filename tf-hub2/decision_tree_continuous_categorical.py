"""
@Project   : DuReader
@Module    : decision_tree_continuous_categorical.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/11/18 3:47 PM
@Desc      : 
"""

"""
Created on Tue Aug  8 21:55:02 2017

@author: zimuliu
"""

"""
https://en.wikipedia.org/wiki/Decision_tree
http://www.cnblogs.com/leoo2sk/archive/2010/09/19/decision-tree.html
https://zhuanlan.zhihu.com/p/28409342

decision tree ID3 algorithm
"""

import numpy as np
from math import log2
from numpy.random import randint, seed

"""
1. function to calculate pk * log2pk
在后面计算信息熵时会用到该函数

求x趋于0时，xlnx的极限
令x=1/y;x（x>0）趋近于0等效为y趋近于正无穷；
xlnx=-lny/y;
令y=e^z;y趋近于正无穷等效于z趋近于正无穷；
xlnx=-lny/y=-z/e^z;
当z趋近于正无穷，e^z的增加要比z的增加快的多，所以z/e^z趋于0
"""


def fun(x):
    if x == 0:
        res = 0
    else:
        res = x * log2(x)
    return res


"""
2. function to calculate information entropy (not infomation gain)
"""


def info(Y):
    # Y是一个一维的numpy array, 长度是初始的长度，但只保留了固定的类别的值
    # 此处Y是二分类值，所以除了p_0就是p_1
    # 在对连续值进行分类时，一般也是做二分类的分割，因此也是除了p_0就是p_1
    p_1 = Y.sum() / Y.size
    # Y中的1所占的比例
    p_0 = 1 - p_1
    # Y中的0所占的比例

    res = - fun(p_0) - fun(p_1)
    # information entropy就是这样定义的
    return res


"""
3. function to calculate conditional information gain
"""


def condition_info(X, Y, split):
    low_rate = (X < split).sum() / X.size
    # X中的元素低于split的比例，后来算加权平均的信息熵时要用到的权重
    high_rate = 1 - low_rate

    low_info = info(Y[np.where(X < split)])
    # X < split所对应的Y的值
    high_info = info(Y[np.where(X >= split)])

    res = low_rate * low_info + high_rate * high_info
    # 加权平均计算分类后的信息熵

    return res


"""
4. function to calculate max gain with different split point of X
"""


def max_split_gain(X, Y):
    X_uniq = np.unique(X)
    n = X_uniq.size
    # 分割点有n种选择
    info_Y = info(Y)
    max_gain = - float('inf')

    for i in range(1, n-1):
        # 从1到n-2，分割点当然不包含最前面和最后面的点
        split = X_uniq[i]
        gain = info_Y - condition_info(X, Y, split)
        # information gain for different splits
        if gain > max_gain:
            max_gain = gain
            max_split = split
        else:
            pass
    return max_split, max_gain


"""
test
"""
seed(10)
X = randint(0, 100, 100)
Y = randint(0, 2, 100)


max_split, max_gain = max_split_gain(X, Y)
print("%.2f" % info(Y))
print("%.2f" % condition_info(X, Y, 18))
print("%d" % max_split, "%.2f" % max_gain)

from sklearn.tree import DecisionTreeClassifier

X2 = np.array([[ele] for ele in X])

clf = DecisionTreeClassifier(criterion="entropy", max_depth=1)
# criterion: “entropy” for the information gain.
# max_depth: The maximum depth of the tree. If None, then nodes are expanded
# until all leaves are pure or until all leaves contain less than
# min_samples_split samples
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.fit

clf.fit(X2, Y)
categories = clf.predict([[16], [17], [18], [19]])
print(categories)
