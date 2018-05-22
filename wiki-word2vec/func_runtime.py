"""
@Project   : DuReader
@Module    : func_runtime.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/22/18 4:03 PM
@Desc      : 
"""
from time import time


def func_runtime(func, n_iter, *args):
    """[summary]

    Arguments:
        func {function object} -- the function to test
        n_iter {int} -- number of iterations

    Returns:
        str -- test result
    """

    start = time()
    for _ in range(n_iter):
        func(*args)
    runtime = (time() - start) / n_iter
    return "Average %.5fs in %d loops" % (runtime, n_iter)
