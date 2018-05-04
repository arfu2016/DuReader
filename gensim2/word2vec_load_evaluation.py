"""
@Project   : DuReader
@Module    : word2vec_evaluation.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/4/18 10:11 AM

@Desc      :
We are publishing pre-trained vectors trained on part of Google News dataset
(about 100 billion words). The model contains 300-dimensional vectors for
3 million words and phrases.
https://code.google.com/archive/p/word2vec/
"""

# shell commands
# !wget https://drive.google.com/file/
# d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

# Data were downloaded in 2018.5

import logging
import os
from gensim.models import KeyedVectors

file_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(file_dir, 'data/GoogleNews-vectors-negative300.bin.gz')

# for logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

model = KeyedVectors.load_word2vec_format(fname, binary=True)
# Load evaluation dataset of analogy task
model.accuracy(os.path.join(file_dir, 'data/questions-words.txt'))
