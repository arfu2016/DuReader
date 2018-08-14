"""
@Project   : text-classification-cnn-rnn
@Module    : group_time.py
@Author    : Deco [deco@cubee.com]
@Created   : 6/21/18 1:49 PM
@Desc      : 
"""
import datetime
import time


if __name__ == '__main__':

    print(type(datetime.datetime.now()))
    print(datetime.datetime.now())

    start = time.time()
    time.sleep(5)
    print(time.time()-start)

    start = time.perf_counter()
    time.sleep(5)
    print(time.perf_counter() - start)
