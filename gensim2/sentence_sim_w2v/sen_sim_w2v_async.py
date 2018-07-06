"""
@Project   : DuReader
@Module    : sen_sim_w2v_async.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/6/18 1:19 PM
@Desc      : 
"""
import os
import time
from multiprocessing import Pool
import logging
import string

import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
from cachetools import cached, TTLCache
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

file_dir = os.path.dirname(os.path.dirname(__file__))
# embed = hub.Module(os.path.join(file_dir,
#                                 'data/universal-sentence-encoder'))
cache = TTLCache(maxsize=100, ttl=300)
# session = tf.Session()
# session.run([tf.global_variables_initializer(), tf.tables_initializer()])
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


# 把句子转换为向量

def model_load():
    fname = os.path.join(file_dir,
                         'data/GoogleNews-vectors-negative300.bin.gz')
    model0 = KeyedVectors.load_word2vec_format(fname, binary=True)
    logging.info('The model was loaded.')
    return model0


def init_model():
    model0 = model_load()
    vocab_dict0 = model0.wv.vocab
    punc_string = string.punctuation + '。，“”‘’（）：；？·—《》、'
    punc_set0 = {punc for punc in punc_string}
    return model0, vocab_dict0, punc_set0


model, vocab_dict, punc_set = init_model()


def _get_word_vector(word):
    # print(type(model.wv[word]))
    return model.wv[word]


class RecordSum:
    record = np.empty(300)

    def add(self, x):
        self.record = np.mean(x, axis=0)


record = RecordSum()


@cached(cache)
def _single_sentence(sentence: str) -> list:
    word_list = word_tokenize(sentence)
    word_list = [word for word in word_list if word not in punc_set]

    word_list = [word for word in word_list if word in vocab_dict]
    with Pool(2) as p:
        # word_vectors = p.map(_get_word_vector, word_list)
        word_vectors = p.map_async(_get_word_vector, word_list,
                                   callback=record.add)
        # print('Type of map result:', type(word_vectors))
        word_vectors.wait()

    # print(record.record)
    st_vector = record.record
    st_vector = st_vector / np.linalg.norm(st_vector)
    st_vector = st_vector.tolist()

    return st_vector


@cached(cache)
def _sentence_embedding(sentences: tuple) -> np.ndarray:
    """use sentences as a tuple is to be consistent with tf.hub"""
    # with Pool(2) as p:
    #     sen_embedding = p.map(_single_sentence, sentences)
    sen_embedding = [_single_sentence(st) for st in sentences]
    sen_embedding = np.array(sen_embedding)

    # embedding = tf.nn.l2_normalize(embed(sentences))
    # sen_embedding = session.run(embedding)

    return sen_embedding


# 给定两个句向量，计算向量相似度

def _vector_similarity(encode1: list, encode2: list) -> float:
    """assume the length of encode1 and encode2 are n, time complexity is
    O(n), space complexity is O(n)
    """
    sim_score = sum([x*y for x, y in zip(encode1, encode2)])

    return sim_score


# 给定一个向量和一组向量，计算前者和后者各个向量的相似度

class VectorSimilarity:
    """Because in multiprocessing package, lambda function can not be pickled.
    This class acts as a helper class to make it pickleable.
    """
    def __init__(self, test_vector):
        self.test_vector = test_vector

    def __call__(self, training_vector):
        return _vector_similarity(training_vector, self.test_vector)


def _similarity_scores(training_vectors: np.ndarray,
                       test_vector: list) -> list:
    """Assume for training vectors, the number of vectors is m, and the
    length of each vector is n, then time complexity is O(mn) for single
    thread. But in numpy, this could be optimized. For multiprocessing, time
    is also reduced.
    """

    training_vectors = training_vectors.tolist()
    test_vector = test_vector

    # sim_scores = [_vector_similarity(training_vector, test_vector)
    #               for training_vector in training_vectors]
    with Pool(2) as p:
        sim_scores = p.map(VectorSimilarity(test_vector),
                           training_vectors)

    return sim_scores


# 给定一个句子和一组句子，找出后者各个句子中和前者最为相似的句子

def most_similar(training_sentences: tuple, test_sentence: str) -> str:
    training_embeddings = _sentence_embedding(training_sentences)
    test_embedding = _single_sentence(test_sentence)
    sim_scores = _similarity_scores(training_embeddings, test_embedding)
    idx = np.argmax(sim_scores)
    return training_sentences[idx]


# test

def t_most_similar():
    start = time.perf_counter()
    print()
    print('Test t_most_similar()')
    training_sentences = ("The quick brown fox jumps over the lazy dog.",
                          "Who is Messy?")
    test_sentence = 'Can you tell me something about Cristiano Ronaldo?'
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'Can you tell me something about Cristiano Ronaldo':")
    print(top_sentence)
    print('time of t_most_similar():', time.perf_counter() - start)


def t2_most_similar():
    start = time.perf_counter()
    print()
    print('Test t2_most_similar()')
    training_sentences = ("The quick brown fox jumps over the lazy dog.",
                          "Who is Messy?")
    test_sentence = 'Something about football.'
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'Something about football':")
    print(top_sentence)
    print('time of t2_most_similar():', time.perf_counter() - start)


if __name__ == '__main__':

    t_most_similar()
    t2_most_similar()

    # session.close()
