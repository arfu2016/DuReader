"""
@Project   : DuReader
@Module    : retrain_new_vacabulary.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/4/18 2:42 PM
@Desc      : 
"""

import logging
import os
import gensim

file_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(file_dir, 'data/word2vec_text8_google.model')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

model = gensim.models.Word2Vec.load(fname)

new_sentences = [['Messi', 'has', 'the', 'edge', 'with', 'intricate', 'skill',
                  'and', 'vision'],
                 ['Ronaldo', 'makes', 'up', 'for', 'with',
                  'strength', 'speed', 'and', 'power', '.'],
                 ['Messi', 'Messi', 'Messi', 'Messi', 'Messi'],
                 ]

model.build_vocab(new_sentences, update=True)
model.train(new_sentences, total_examples=model.corpus_count, epochs=3)
