"""
@Project   : DuReader
@Module    : word2vec_tutorial.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/2/18 5:20 PM
@Desc      : https://rare-technologies.com/word2vec-tutorial/
"""

# import modules & set up logging

import logging
import os
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


sentences = MySentences('/some/directory')  # a memory-friendly iterator
model_too = gensim.models.Word2Vec(sentences)

# Evaluating
# Google have released their testing set of about 20,000 syntactic and semantic
# test examples, following the “A is to B as C is to D” task:
# https://raw.githubusercontent.com/RaRe-
# Technologies/gensim/develop/gensim/test/test_data/questions-words.txt.
model.accuracy('/tmp/questions-words.txt')
