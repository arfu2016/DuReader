"""
@Project   : DuReader
@Module    : module_test.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/14/18 1:51 PM
@Desc      : 
"""
import os
import sys

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from mysql2.mysql_settings import test, save_data

if __name__ == "__main__":
    # test()
    save_data()
