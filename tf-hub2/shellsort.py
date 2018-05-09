"""
@Project   : DuReader
@Module    : shellsort.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/9/18 1:16 PM
@Desc      :
希尔排序是小规模数据排序的最优选择，算法的核心思想就是对插入排序的封装，
逐渐减小增量形成新的数组进行插入排序。试着用Python实现了希尔排序，并测试通过。
测试方法：

测试1000组数据
生成随机数列表A，包含100个属于[-5, 5]的随机数
拷贝A的元素至列表B
列表A使用希尔排序
列表B使用系统自带的排序方法
对比列表A和B
如果正确的次数等于1000则通过测试
"""
import random
import copy
import operator

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 21:45:04 2017

@author: liuzimu
"""


def shell_sort(A):
    N = len(A)
    increment = N//2

    while increment > 0:
        i = increment

        while i < N:
            j = i - increment
            tmp = A[i]

            while j >= 0 and A[j] > tmp:
                A[j + increment] = A[j]
                j -= increment

            A[j + increment] = tmp
            i += 1

        increment //= 2

    return None


"""
test
"""

k = 0
res = 0
t = 1000

while k < t:
    A = [random.randint(-5, 5) for x in range(100)]
    B = copy.copy(A)

    shell_sort(A)
    B.sort()

    res += int(operator.eq(A, B))
    k += 1

if res == t:
    print("Pass!")
else:
    print("Wrong!")
