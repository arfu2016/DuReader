"""
@Project   : DuReader
@Module    : shortcuts.py
@Author    : Deco [deco@cubee.com]
@Created   : 4/9/18 4:18 PM
@Desc      : 
"""
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from tensorflow.run_python3 import run

if __name__ == '__main__':
    run()

# python ~/projects/DuReader/run.py --train --algo MLSTM --epochs 2 --batch_size 16
