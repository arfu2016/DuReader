"""
@Project   : DuReader
@Module    : zh_vector_evaluation.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/7/18 2:00 PM
@Desc      : 
"""

import gensim
# import pprint

model = gensim.models.Word2Vec.load("data/wiki.zh.model")
print('The model was loaded.')

word = "足球"
results = model.most_similar(word)
print('Similar words for {}'.format(word))
for result in results:
    print(result[0], result[1])
print()

word = "巴塞罗那"
results = model.most_similar(word)
print('Similar words for {}'.format(word))
for result in results:
    print(result[0], result[1])
print()

word = "赛程"
results = model.most_similar(word)
print('Similar words for {}'.format(word))
for result in results:
    print(result[0], result[1])
print()
