"""
@Project   : DuReader
@Module    : train_val_split.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/21/18 5:36 PM
@Desc      : 
"""

from sklearn.model_selection import train_test_split


def train_val_test(ti):
    train_xy, val_xy = train_test_split(ti, test_size=0.25, random_state=0)
    val_xy, test_xy = train_test_split(val_xy, test_size=0.25, random_state=0)
    return train_xy, val_xy, test_xy
