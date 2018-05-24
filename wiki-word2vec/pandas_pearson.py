"""
@Project   : DuReader
@Module    : pandas_pearson.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/24/18 11:52 AM
@Desc      :
https://github.com/mediaProduct2017/dataAnalysis/blob/master/1p4ETL_SQL.ipynb
"""

print(ttc_db[['Total_Tests_Completed', 'Mean_ITI_minutes']].corr(
    method='pearson', min_periods=1))
