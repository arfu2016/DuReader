"""
@Project   : DuReader
@Module    : linear_regression.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/17/18 11:20 AM
@Desc      : 
"""

"""
Created on Thu Aug 24 19:51:30 2017

@author: zimuliu
"""

"""
https://en.wikipedia.org/wiki/Simple_linear_regression
https://en.wikipedia.org/wiki/Stochastic_gradient_descent
"""

"""
0. initialization
"""

import numpy as np
from numpy.random import randint
from matplotlib import pyplot as plt

"""
1. define least square estimation function: y = ax + b
"""


def lse(X, Y):

    # cov_mat = np.cov(X, Y, ddof=0)
    # a = cov_mat[0][1] / cov_mat[0][0]

    Y_mean = Y.mean()
    X_mean = X.mean()
    a = np.sum(np.multiply(X, Y-Y_mean))/np.sum(np.multiply(X, X-X_mean))
    # 使用求导的方法得到解析式
    b = Y_mean - a * X_mean

    Y_pred = a * X + b
    Rsquare = np.mean((Y_pred - Y_mean) ** 2) / np.var(Y)

    return a, b, Rsquare


"""
2. define stochastic gradient descent function: y = ax + b
n_samples [0,1]
alpha (0,1]
"""


def sgd(X, Y, alpha=0.0001, n_samples=1):
    n = X.size
    n_samples = int(n_samples * n)

    a = 0
    b = 0

    for i in range(n_samples):
        j = randint(low=0, high=n)
        # mini-batch gradient descent
        # stochastic gradient descent
        # batch size=n_samples, epochs=1
        # or we can regard, batch sieze=1, epochs=n_samples
        # online training

        tmp = alpha * 2 * (Y[j] - (a * X[j] + b))

        a += X[j] * tmp
        b += tmp

    Y_pred = a * X + b
    Rsquare = np.mean((Y_pred - Y.mean()) ** 2) / np.var(Y)

    return a, b, Rsquare


"""
3. generate X, Y with linear relation then add random noise
"""


def random_xy(noise=5):
    np.random.seed(0)
    X = np.array([randint(low=0, high=100) for x in range(100)])

    Y = np.array([x * 3 + randint(low=- noise, high=noise) for x in X])
    # Y = np.array([x * 3 for x in X])

    return X, Y


"""
4. calculate slope and intercept:
red line: least square estimation
blue line: stochastic gradient descent
"""


def main():
    X, Y = random_xy(noise=60)

    a, b, r1 = lse(X, Y)
    c, d, r2 = sgd(X, Y, alpha=10e-6, n_samples=0.6)

    plt.scatter(X, Y, s=5, color='k')
    plt.plot(X, a * X + b, color='r')
    plt.plot(X, c * X + d, color='b')

    plt.title("Linear regression")
    plt.legend(['lse R^2 is ' + str(r1)[:5], 'sgd R^2 is ' + str(r2)[:5]])

    plt.show()


"""
5. main
"""
if __name__ == '__main__':
    main()
