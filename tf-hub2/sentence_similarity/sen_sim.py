"""
@Project   : DuReader
@Module    : sen_sim.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/2/18 10:37 AM
@Desc      : 给出一组句子，找到其中和一个另给的句子最相似的句子
"""
import os
import time
from functools import reduce
from multiprocessing import Pool
from threading import RLock

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from cachetools import cached, TTLCache

file_dir = os.path.dirname(os.path.dirname(__file__))
embed = hub.Module(os.path.join(file_dir,
                                'data/universal-sentence-encoder'))
# embed = hub.Module("https://tfhub.dev/google/"
#                    "universal-sentence-encoder/1")

cache = TTLCache(maxsize=100, ttl=300)
# If no expired items are there to remove, the least recently used items will
# be discarded first to make space when necessary.

lock = RLock()

# 把句子转换为向量

# 某个函数值如果从来不变，可以用cache，也可以作为module attribute或者class attribute保存
# 某个函数值固定时间更新（比如每天更新一次），可以用cache
# 某个函数值每次都变，不要用cache，每次都直接调用函数

session = tf.Session()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])


@cached(cache)
def _sentence_embedding(sentences: tuple) -> np.ndarray:
    embedding = tf.nn.l2_normalize(embed(sentences))
    # embedding = embed(sentences)
    sen_embedding = session.run(embedding)

    return sen_embedding


def _produce_sentence_embedding():
    training_sentences = ("The quick brown fox jumps over the lazy dog.",
                          "Who is Messy")
    training_embeddings = _sentence_embedding(training_sentences)

    test_sentence = ('Can you tell me something about Messy',)
    test_embedding = _sentence_embedding(test_sentence)

    return training_embeddings, test_embedding


def test_sentence_embedding():
    print()
    print('Test test_sentence_embedding()')
    training_embeddings, test_embedding = _produce_sentence_embedding()

    print('Shape of embeddings of training sentences:')
    print(training_embeddings.shape)
    print('Shape of embeddings of test sentence:')
    print(test_embedding.shape)


def test_cache():
    print()
    print('Test test_cache()')
    with lock:
        cache.clear()
    # to test the effect of cache
    # synchronization, 使用lock主要为了应对多线程情况
    # [clear the cache](http://cachetools.readthedocs.io/en/latest/)
    start = time.perf_counter()
    _produce_sentence_embedding()
    print('time without cache:', time.perf_counter()-start)
    start = time.perf_counter()
    _produce_sentence_embedding()
    print('time with cache:', time.perf_counter() - start)


# 给定两个句向量，计算向量相似度


def multiplication_two_number(tuple_two_numbers):
    return tuple_two_numbers[0]*tuple_two_numbers[1]


def _multiplication(encode1: list, encode2: list):
    for x, y in zip(encode1, encode2):
        yield x*y
    # p = Pool(2)
    # res = p.imap(multiplication_two_number, zip(encode1, encode2))
    # # res = p.imap(lambda x, y: x*y, zip(encode1, encode2))
    # p.close()
    # p.join()
    # return res


def test_multiplication():
    print()
    print('Test test_multiplication()')
    print('Result of _multiplication:')
    print(_multiplication([0, 1], [1, 2]))


def add(x, y):
    return x+y


def _vector_similarity(encode1: list, encode2: list) -> float:
    """assume the length of encode1 and encode2 are n, time complexity is
    O(n), space complexity is O(n)
    """
    sim_score = reduce(add, _multiplication(encode1, encode2))
    # sim_score = sum(_multiplication(encode1, encode2))
    # sim_score = sum([x*y for x, y in zip(encode1, encode2)])
    # sim_score = np.sum(np.multiply(encode1, encode2))
    # realization of dot product of two vectors

    return sim_score


def test_vector_similarity():
    print()
    print('Test test_vector_similarity()')
    print('Result of _vector_similarity:')
    print(_vector_similarity([0, 1], [1, 2]))


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
    length of each vector is n, then time complexity is O(mn). But in numpy,
    this could be optimized.
    """

    training_vectors = training_vectors.tolist()
    test_vector = test_vector.tolist()
    test_vector = test_vector[0]

    with Pool(2) as p:
        sim_scores = p.map(VectorSimilarity(test_vector),
                           training_vectors)

        # sim_scores = p.map_async(VectorSimilarity(test_vector),
        #                          training_vectors)
        # map的异步调用，返回的iterator中没有顺序，类似一个集合

        # sim_scores = p.map(lambda x: _vector_similarity(x, test_vector),
        #                    training_vectors)

    # sim_scores = [_vector_similarity(training_vector, test_vector)
    #               for training_vector in training_vectors]

    return sim_scores


def test_similarity_scores():
    print()
    print('Test test_similarity_scores()')
    training_embeddings, test_embedding = _produce_sentence_embedding()
    sim_scores = _similarity_scores(training_embeddings, test_embedding)
    print('type of similarity scores:')
    print(type(sim_scores))
    print('Similarity scores:')
    print(sim_scores)


# 给定一个句子和一组句子，找出后者各个句子中和前者最为相似的句子

def most_similar(training_sentences: tuple, test_sentence: tuple) -> str:
    start = time.perf_counter()
    training_embeddings = _sentence_embedding(training_sentences)
    print('time of calculating training_embeddings:',
          time.perf_counter() - start)
    start = time.perf_counter()
    test_embedding = _sentence_embedding(test_sentence)
    print('time of calculating test_embedding:',
          time.perf_counter() - start)
    sim_scores = _similarity_scores(training_embeddings, test_embedding)
    idx = np.argmax(sim_scores)
    return training_sentences[idx]


def test_most_similar():
    start = time.perf_counter()
    print()
    print('Test test_most_similar()')
    training_sentences = ("The quick brown fox jumps over the lazy dog.",
                          "Who is Messy")
    test_sentence = ('Can you tell me something about Cristiano Ronaldo',)
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'Can you tell me something about Cristiano Ronaldo':")
    print(top_sentence)
    print('time of test_most_similar():', time.perf_counter() - start)


if __name__ == '__main__':
    test_multiplication()
    test_vector_similarity()

    test_sentence_embedding()
    test_cache()
    test_similarity_scores()
    test_most_similar()

    session.close()
