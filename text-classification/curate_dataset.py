"""
@Project   : DuReader
@Module    : curate_dataset.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/17/18 5:49 PM
@Desc      : 
"""

g = open('data/reviews.txt', 'r')  # What we know!
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()
# map函数就就是并行计算中的map操作，g是一个iterator，map操作可以作用于每一行，把最后的换行符去掉

g = open('data/labels.txt', 'r')  # What we WANT to know!
labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
g.close()

print(len(reviews))
print(reviews[0])
