"""
@Project   : DuReader
@Module    : word2vec_merge_google.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/4/18 1:34 PM
@Desc      : improve word vectors by google vectors
"""
import logging
import os
import gensim
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

sentences = word2vec.Text8Corpus("data/text8")

model = gensim.models.Word2Vec(sentences, size=300, min_count=5, iter=1)

file_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(file_dir, 'data/GoogleNews-vectors-negative300.bin.gz')

model.intersect_word2vec_format(fname,
                                lockf=1.0,
                                binary=True)

model.train(sentences, total_examples=model.corpus_count, epochs=5)
model.save(os.path.join(file_dir, 'data/word2vec_text8_google.model'))
