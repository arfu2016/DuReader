"""
@Project   : DuReader
@Module    : train_word2vec_model.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/7/18 12:53 PM
@Desc      :
python train_word2vec_model.py data/wiki.zh.text.seg
data/wiki.zh.model data/wiki.zh.vector

%(program)s has a problem
"""

# train_word2vec_model.py用于训练模型

import logging
import os.path
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) != 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    inp, outp, outp2 = sys.argv[1:4]

    model = Word2Vec(LineSentence(inp), size=300, window=5, iter=5,
                     min_count=5, workers=multiprocessing.cpu_count())

    model.save(outp)
    model.wv.save_word2vec_format(outp2, binary=False)
