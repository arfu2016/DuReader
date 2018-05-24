"""
@Project   : DuReader
@Module    : feature_scaling.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/24/18 11:54 AM
@Desc      : 
"""
from sklearn.preprocessing import scale
# scale is used for feature scaling
import numpy as np


X = ttc_db[['Total_Tests_Completed', 'Mean_ITI_minutes']].dropna(axis=0)
# axis=0 means drop the rows with np.NaN, axis=1 means drop the columns

print(np.cov(np.array(scale(X)).transpose()))
