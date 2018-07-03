"""
@Project   : DuReader
@Module    : sen_sim.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/2/18 10:37 AM
@Desc      : 给出一组句子，找到其中和一个另给的句子最相似的句子
"""
import os
import time
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

# 1. 把句子转换为向量


@cached(cache)
def _sentence_embedding(sentences: tuple) -> np.ndarray:
    embedding = tf.nn.l2_normalize(embed(sentences))

    with tf.Session() as session:
        session.run(
            [tf.global_variables_initializer(), tf.tables_initializer()])
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
    # synchronization, 使用lock主要为了应对多线程情况
    # [clear the cache](http://cachetools.readthedocs.io/en/latest/)
    # 多进程实现的话一般是异步实现？
    start = time.perf_counter()
    _produce_sentence_embedding()
    print('time without cache:', time.perf_counter()-start)
    start = time.perf_counter()
    _produce_sentence_embedding()
    print('time with cache:', time.perf_counter() - start)


# 2. 给定两个句向量，计算向量相似度

def _vector_similarity(encode1: list, encode2: list) -> float:

    sim_score = np.sum(np.multiply(encode1, encode2))
    # realization of dot product of two vectors

    return sim_score


# 3. 给定一个向量和一组向量，计算前者和后者各个向量的相似度

class VectorSimilarity:
    def __init__(self, test_vector):
        self.test_vector = test_vector

    def __call__(self, training_vector):
        return _vector_similarity(training_vector, self.test_vector)


def _similarity_scores(training_vectors: np.ndarray,
                       test_vector: np.ndarray) -> list:

    training_vectors = training_vectors.tolist()
    test_vector = test_vector.tolist()
    test_vector = test_vector[0]

    # sim_scores = [_vector_similarity(training_vector, test_vector)
    #               for training_vector in training_vectors]

    with Pool(2) as p:
        # sim_scores = p.map(lambda x: _vector_similarity(x, test_vector),
        #                    training_vectors)
        sim_scores = p.map(VectorSimilarity(test_vector), training_vectors)
    return sim_scores


def test_similarity_scores():
    print()
    print('Test test_similarity_scores()')
    training_embeddings, test_embedding = _produce_sentence_embedding()
    sim_scores = _similarity_scores(training_embeddings, test_embedding)
    print('Similarity scores:')
    print(sim_scores)


# 4. 给定一个句子和一组句子，找出后者各个句子中和前者最为相似的句子

def most_similar(training_sentences: tuple, test_sentence: tuple) -> str:
    training_embeddings = _sentence_embedding(training_sentences)
    test_embedding = _sentence_embedding(test_sentence)
    sim_scores = _similarity_scores(training_embeddings, test_embedding)
    idx = np.argmax(sim_scores)
    return training_sentences[idx]


def test_most_similar():
    print()
    print('Test test_most_similar()')
    training_sentences = ("The quick brown fox jumps over the lazy dog.",
                          "Who is Messy")
    test_sentence = ('Can you tell me something about Cristiano Ronaldo',)
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'Can you tell me something about Cristiano Ronaldo':")
    print(top_sentence)


if __name__ == '__main__':
    test_sentence_embedding()
    test_cache()
    test_similarity_scores()
    test_most_similar()
