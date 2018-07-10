"""
@Project   : DuReader
@Module    : sen_sim_tfhub.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/5/18 3:05 PM
@Desc      : 给出一组句子，找到其中和一个另给的句子最相似的句子
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from cachetools import cached, TTLCache

file_dir = os.path.dirname(os.path.dirname(__file__))
embed = hub.Module(os.path.join(file_dir,
                                'data/universal-sentence-encoder'))
cache = TTLCache(maxsize=100, ttl=300)
session = tf.Session()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])


# 把句子转换为向量

@cached(cache)
def _sentence_embedding(sentences: tuple) -> np.ndarray:
    embedding = tf.nn.l2_normalize(embed(sentences))
    sen_embedding = session.run(embedding)

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
                       test_vector: np.ndarray) -> list:
    """Assume for training vectors, the number of vectors is m, and the
    length of each vector is n, then time complexity is O(mn) for single
    thread. But in numpy, this could be optimized. For multiprocessing, time
    is also reduced.
    """

    training_vectors = training_vectors.tolist()
    test_vector = test_vector.tolist()
    test_vector = test_vector[0]

    with Pool(2) as p:
        sim_scores = p.map(VectorSimilarity(test_vector),
                           training_vectors)

    return sim_scores


# 给定一个句子和一组句子，找出后者各个句子中和前者最为相似的句子

def most_similar(training_sentences: tuple, test_sentence: tuple) -> str:
    training_embeddings = _sentence_embedding(training_sentences)
    start = time.perf_counter()
    test_embedding = _sentence_embedding(test_sentence)
    print('time to get sentence vector:', time.perf_counter()-start)
    sim_scores = _similarity_scores(training_embeddings, test_embedding)
    print('sim_scores:', sim_scores)
    idx = np.argmax(sim_scores)
    return training_sentences[idx]


# test

def t_most_similar():
    start = time.perf_counter()
    print()
    print('Test t_most_similar()')
    training_sentences = ("The quick brown fox jumps over the lazy dog",
                          "Who is Messy")
    test_sentence = ('Can you tell me something about Cristiano Ronaldo',)
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'Can you tell me something about Cristiano Ronaldo':")
    print(top_sentence)
    print('time of t_most_similar():', time.perf_counter() - start)


def t2_most_similar():
    start = time.perf_counter()
    print()
    print('Test t2_most_similar()')
    training_sentences = ("The quick brown fox jumps over the lazy dog",
                          "Who is Messy")
    test_sentence = ('Something about football',)
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'Something about football':")
    print(top_sentence)
    print('time of t2_most_similar():', time.perf_counter() - start)


if __name__ == '__main__':

    t_most_similar()
    t2_most_similar()

    session.close()
