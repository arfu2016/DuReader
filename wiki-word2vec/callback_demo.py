"""
@Project   : DuReader
@Module    : callback_demo.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/23/18 2:23 PM
@Desc      :
http://code.activestate.com/recipes/580787-implementing-function-based-callbacks-in-python/
"""

# File: callback_demo.py
# To demonstrate implementation and use of callbacks in Python,
# using just plain functions.
# Author: Vasudev Ram
# Copyright 2017 Vasudev Ram
# Web site: https://vasudevram.github.io
# Blog: https://jugad2.blogspot.com
# Product store: https://gumroad.com/vasudevram

from __future__ import print_function
from time import sleep


def callback_a(i, result):
    print("Items processed: {}. Running result: {}.".format(i, result))


def square(i):
    return i * i


def processor(process, times, report_interval, callback):
    print(
        "Entered processor(): times = {}, report_interval = {}, callback = {}".format(
            times, report_interval, callback.__name__))
    # Can also use callback.__name__ instead of callback.func_name
    # in line above.
    result = 0
    print("Processing data ...")
    for i in range(1, times + 1):
        result += process(i)
        sleep(1)
        if i % report_interval == 0:
            # This is the call to the callback function
            # that was passed to this function.
            callback(i, result)


processor(square, 20, 5, callback_a)
