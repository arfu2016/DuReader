"""
@Project   : DuReader
@Module    : run_python3.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/23/18 5:47 PM
@Desc      : 
"""

import os
import sys

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
# 以base_dir为基准开始导入

from mreading.dataset import BRCDataset
