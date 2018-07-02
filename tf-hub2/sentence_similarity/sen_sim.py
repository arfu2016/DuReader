"""
@Project   : DuReader
@Module    : sen_sim.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/2/18 10:37 AM
@Desc      : 
"""
import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
from cachetools import cached, TTLCache
import time
from threading import RLock

# embed = hub.Module("https://tfhub.dev/google/"
#                    "universal-sentence-encoder/1")

file_dir = os.path.dirname(os.path.dirname(__file__))
embed = hub.Module(os.path.join(file_dir,
                                'data/universal-sentence-encoder'))

cache = TTLCache(maxsize=100, ttl=300)
# If no expired items are there to remove, the least recently used items will
# be discarded first to make space when necessary.

lock = RLock()


@cached(cache)
def _sentence_embedding(sentences: tuple) -> np.ndarray:
    embedding = tf.nn.l2_normalize(embed(sentences))

    with tf.Session() as session:
        session.run(
            [tf.global_variables_initializer(), tf.tables_initializer()])
        sen_embedding = session.run(embedding)

    # time.sleep(0.1)
    return sen_embedding


def _produce_sentence_embedding():
    training_sentences = ("The quick brown fox jumps over the lazy dog.",
                          "Who is Messy")
    training_embeddings = _sentence_embedding(training_sentences)
    # print('type of training_embeddings:')
    # print(type(training_embeddings))

    test_sentence = ('Can you tell me something about Messy',)
    test_embedding = _sentence_embedding(test_sentence)

    # print('Embeddings of training sentences:')
    # print(training_embeddings)
    # print('Embeddings of test sentence:')
    # print(test_embedding)
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


def _vector_similarity(encode1: list, encode2: list) -> float:

    sim_score = np.sum(np.multiply(encode1, encode2))
    # realization of dot product of two vectors

    return sim_score


def similarity_scores(training_vectors: np.ndarray,
                      test_vector: np.ndarray) -> list:

    training_vectors = training_vectors.tolist()
    test_vector = test_vector.tolist()
    test_vector = test_vector[0]

    sim_scores = [_vector_similarity(training_vector, test_vector)
                  for training_vector in training_vectors]

    return sim_scores


def test_similarity_scores():
    print()
    print('Test test_similarity_scores()')
    training_embeddings, test_embedding = _produce_sentence_embedding()
    sim_scores = similarity_scores(training_embeddings, test_embedding)
    print('Similarity scores:')
    print(sim_scores)


def most_similar(training_sentences: list, test_sentence: list) -> str:
    pass


if __name__ == '__main__':
    test_sentence_embedding()
    test_cache()
    test_similarity_scores()
