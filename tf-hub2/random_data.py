"""
@Project   : DuReader
@Module    : random_data.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/11/18 3:33 PM
@Desc      : 
"""

from numpy.random import (randint, seed)
import random

seed(10)
X = randint(0, 100, 100)
Y = randint(0, 2, 100)

random.seed(0)
Z = random.random()

print('X:', X)
print('Y:', Y)
print('Z:', Z)
