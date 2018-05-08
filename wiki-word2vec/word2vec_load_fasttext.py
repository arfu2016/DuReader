"""
@Project   : DuReader
@Module    : word2vec_evaluation.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/4/18 10:11 AM

@Desc      :
We are publishing pre-trained word vectors for 294 languages, trained on
Wikipedia using fastText. These vectors in dimension 300 were obtained using
the skip-gram model described in Bojanowski et al. (2016) with default
parameters.
The dataset has about 300,000 words
https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
https://fasttext.cc/docs/en/crawl-vectors.html
"""

# Data were downloaded in 2017

import logging
import os
import sys
from gensim.models import KeyedVectors

file_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(file_dir, 'data/wiki.zh.vec')

# for logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

model = KeyedVectors.load_word2vec_format(fname, binary=False)
# Load evaluation dataset of analogy task

logger.info('The model was loaded.')

word = "足球"
results = model.most_similar(word)
logger.info('Similar words for {}'.format(word))
for result in results:
    print(result[0], result[1])
print()

word = "巴塞罗那"
try:
    results = model.most_similar(word)
    logger.info('Similar words for {}'.format(word))
    for result in results:
        print(result[0], result[1])
except KeyError:
    logger.info('{} not in vocabulary'.format(word))
print()

word = "赛程"
try:
    results = model.most_similar(word)
    logger.info('Similar words for {}'.format(word))
    for result in results:
        print(result[0], result[1])
except KeyError:
    logger.info('{} not in vocabulary'.format(word))
print()
