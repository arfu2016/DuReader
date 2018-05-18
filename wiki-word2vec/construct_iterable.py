"""
@Project   : DuReader
@Module    : construct_iterable.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/18/18 3:36 PM
@Desc      : 
"""
import os
import gensim


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


sentences = MySentences('/some/directory')  # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences)
