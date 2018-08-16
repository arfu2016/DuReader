"""
@Project   : DuReader
@Module    : text8_fit.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/3/18 4:04 PM

@Desc      :
text8 text file
http://mattmahoney.net/dc/textdata.html
Wikipedia text (enwik9, 1 GB)
equivalent cleaned text (fil9, 715 MB)
text8 is the first 10^8 bytes of fil9.
"""

# shell commands
# !wget http://mattmahoney.net/dc/text8.zip

# unzip the file

import logging
import os
import gensim
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

file_name = os.path.join(base_dir, "gensim2/data/text8")

sentences = word2vec.Text8Corpus(file_name)

model = gensim.models.Word2Vec(sentences, size=300, min_count=5, iter=6)

fname = os.path.join(base_dir, 'gensim2/data/word2vec_text8.model')
model.save(fname)
