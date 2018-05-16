"""
@Project   : DuReader
@Module    : binary_search.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/16/18 10:50 AM
@Desc      : 
"""
from bisect import bisect_left


def index(a, x):
    """binary search: Locate the leftmost value exactly equal to x;
    search list a for x"""
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError
