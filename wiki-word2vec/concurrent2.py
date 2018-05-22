"""
@Project   : DuReader
@Module    : concurrent.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/22/18 3:34 PM
@Desc      : 
"""
"""
Python并发编程
https://zhuanlan.zhihu.com/p/36745253
https://zhuanlan.zhihu.com/p/25377631

Python的并发编程有多个包可以选择，如multiprocessing，threading。然而，最终还是选择了
concurrent.futures，ProcessPoolExecutor表示多进程, ThreadPoolExecutor表示多线程。
注意：一定要在linux上测试，windows的多进程还不如单进程快。windows开新的进程开销太大
"""

"""
@Author: liuzimu 
@Date: 2018-05-11 20:45:53 
@Last Modified by: liutienan 
@Last Modified time: 2018-05-11 20:45:53 
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from numpy.random import randint
from time import time


def find(sub_str, target_str):
    """[summary]
    字符串查找

    Arguments:
        sub_str {str} -- substring
        target_str {str} -- target string

    Returns:
        bool -- if substring is found in target string
    """

    return sub_str in target_str
    # in operator用的是Boyer–Moore算法，最坏情况O(mn), 最好情况O(n/m)
    # 速度肯定好过re.search()


def rand_str(size):
    """[summary]
    随机字符串

    Arguments:
        size {int} -- size of string

    Returns:
        str -- string with random characters of A-Z
    """

    return ''.join(map(chr, randint(low=65, high=90, size=size)))
    # chr: Return a string of one character whose ASCII code is the integer i


def serial_func(sub_strs, target_str):
    """[summary]
    串行: 在一个字符串中寻找多个子串，用for循环实现

    Arguments:
        sub_strs {list} -- a list of substrings
        target_str {str} -- target string

    Returns:
        str -- a sub string found in target string
    """

    for sub_str in sub_strs:
        if find(sub_str, target_str):
            return sub_str


def parallel_func(sub_strs, target_str, executor):
    """[summary]
    并行: 在一个字符串中寻找多个子串，用并发技术实现
    多进程和多cpu、gpu编程是类似的
    hadoop、spark是多进程的自动化？特别是多进程操作I/O的自动化?
    python在一个cpu上的多线程不是真正的同时运行的多线程，本质上还是单线程，只是经过了优化
    python在一个cpu上的多进程也不是真正的同时运行的多进程，虽然有多个Python process在运行，
    但并不是同时的，还是依次在cpu上运行，只是经过了优化
    python在多个cpu（或多个核）上的多进程是真正的多进程，一般每个cpu核上跑一个进程

    Arguments:
        sub_strs {list} -- a list of substrings
        target_str {str} -- target string
        executor {Executor} -- existance of ProcessPoolExecutor, ThreadPoolExecutor

    Returns:
        str -- a sub string found in target string
    """

    return executor.submit(serial_func, sub_strs, target_str)


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


if __name__ == "__main__":
    # generate target string and substring list
    target_str = rand_str(1000)
    n = 1200
    sub_strs = [rand_str(10) for _ in range(n)]
    # get executor existance
    multi_process = ProcessPoolExecutor()
    multi_thread = ThreadPoolExecutor()
    # how many times to execute test functions
    n_iter = 10000

    # print test results
    print(func_runtime(serial_func, n_iter, sub_strs, target_str))
    print(func_runtime(parallel_func, n_iter, sub_strs, target_str,
                       multi_process))
    print(func_runtime(parallel_func, n_iter, sub_strs, target_str,
                       multi_thread))
