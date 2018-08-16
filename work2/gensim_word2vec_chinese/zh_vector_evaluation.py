"""
@Project   : DuReader
@Module    : zh_vector_evaluation.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/7/18 2:00 PM
@Desc      : 
"""
import os
import gensim
# import pprint

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

file_name = os.path.join(base_dir, "gensim2/data/wiki.zh.model")

model = gensim.models.Word2Vec.load(file_name)
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
