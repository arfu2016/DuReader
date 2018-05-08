"""
@Project   : DuReader
@Module    : word2vec_vocab.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/8/18 5:03 PM
@Desc      : 
"""
import gensim


def model_load():
    model0 = gensim.models.Word2Vec.load("data/wiki.zh.model")
    print('The model was loaded.')
    return model0


model = model_load()
vocab_dict = model.wv.vocab
vocab_list = vocab_dict.keys()
print('Number of words:', len(vocab_dict))
print('Type of vector:', type(model.wv['哪些']))

i = 0
words = []
for word in vocab_list:
    i += 1
    words.append(word)
    if i == 10:
        break
print('Examples of words:', words)
