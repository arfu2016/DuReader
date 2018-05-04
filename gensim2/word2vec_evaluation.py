"""
@Project   : DuReader
@Module    : word2vec_evaluation.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/4/18 11:07 AM

@Desc      :
Analogy task for word2vec evaluation
Word2vec training is an unsupervised task, there’s no good way to objectively
evaluate the result. Evaluation depends on your end application.

Google have released their testing set of about 20,000 syntactic and semantic
test examples, following the “A is to B as C is to D” task:
https://raw.githubusercontent.com/RaRe-Technologies/gensim/develop/gensim/test
/test_data/questions-words.txt.
"""

import logging
import os
import gensim

file_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(file_dir, 'data/word2vec_text8_google.model')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

model = gensim.models.Word2Vec.load(fname)

# Load evaluation dataset of analogy task
model.accuracy(os.path.join(file_dir, 'data/questions-words.txt'))
