"""
@Project   : DuReader
@Module    : sqrt.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/11/18 9:47 AM
@Desc      : 
"""

"""
Created on Wed Aug  2 20:22:25 2017

@author: liuzimu
"""

"""
https://www.zhihu.com/question/35133069
@Haojun
"""

"""
binary method, n is positive integer
"""


def binary_method(n, threshold):
    left, right = 0, n

    while 1:
        mid = (left + right) / 2
        sqr = mid ** 2

        if abs(sqr - n) <= threshold:
            break
        elif sqr > n:
            right = mid
        else:
            left = mid

    return mid


"""
newton method, n is positive integer

Theory:

牛顿法求解一元高次方程

slope of the tangent line is f'(Xn)

设切线方程为y=ax+b
a已经知道，是切线斜率
知道切线过一点(Xn, f(Xn))，带入切线方程中可以求出b
这样切线方程就求出来了，切线在x轴上的交点为(Xn+1, 0)
这个点带入切线方程，就可以得到下面的等式

[Xn, f(Xn)] is one point of the tangent line
f(Xn) - 0 = f'(Xn) * (Xn - Xn+1)
Xn+1 = Xn - f(Xn) / f'(Xn)
f(x) = x ** 2 - n
f'(x) = 2 * x
Xn+1 = Xn - (Xn - n / Xn) / 2
Xn+1 = (Xn + n / Xn) / 2

"""


def newton_method(n, threshold):
    res = n

    while abs(res ** 2 - n) > threshold:
        res = (res + n / res) / 2

    return res


print("%.3f" % binary_method(2, 10e-5))
print("%.3f" % newton_method(2, 10e-5))

print("%.3f" % 2**0.5)

'''
%timeit binary_method(2, 10e-5)
%timeit newton_method(2, 10e-5)
'''
