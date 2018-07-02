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

# embed = hub.Module("https://tfhub.dev/google/"
#                    "universal-sentence-encoder/1")

file_dir = os.path.dirname(os.path.dirname(__file__))
embed = hub.Module(os.path.join(file_dir,
                                'data/universal-sentence-encoder'))

cache = TTLCache(maxsize=100, ttl=300)
# If no expired items are there to remove, the least recently used items will
# be discarded first to make space when necessary.


@cached(cache)
def sentence_embedding(sentences: tuple) -> np.ndarray:
    embedding = tf.nn.l2_normalize(embed(sentences))

    with tf.Session() as session:
        session.run(
            [tf.global_variables_initializer(), tf.tables_initializer()])
        sen_embedding = session.run(embedding)
    return sen_embedding


def test_sentence_embedding():
    training_sentences = ("The quick brown fox jumps over the lazy dog.",
                          "Who is Messy")
    training_embeddings = sentence_embedding(training_sentences)
    # print('type of training_embeddings:')
    # print(type(training_embeddings))

    test_sentence = ('Can you tell me something about Messy',)
    test_embedding = sentence_embedding(test_sentence)

    # print('Embeddings of training sentences:')
    # print(training_embeddings)
    # print('Embeddings of test sentence:')
    # print(test_embedding)

    return training_embeddings, test_embedding


def test_cache():
    start = time.perf_counter()
    test_sentence_embedding()
    print('time without cache:', time.perf_counter()-start)
    start = time.perf_counter()
    test_sentence_embedding()
    print('time after cache:', time.perf_counter() - start)


def vector_similarity(encode1: list, encode2: list) -> float:

    sim_scores = np.sum(np.multiply(encode1, encode2))
    # realization of dot product of two vectors

    return sim_scores


def similarity_scores(training_vectors: np.ndarray,
                      test_vector: np.ndarray) -> list:

    training_vectors = training_vectors.tolist()
    test_vector = test_vector.tolist()
    test_vector = test_vector[0]

    sim_scores = [vector_similarity(training_vector, test_vector)
                  for training_vector in training_vectors]

    return sim_scores


def test_similarity_scores():
    training_embeddings, test_embedding = test_sentence_embedding()
    sim_scores = similarity_scores(training_embeddings, test_embedding)
    print('Similarity scores:')
    print(sim_scores)


if __name__ == '__main__':
    test_sentence_embedding()
    test_cache()
    test_similarity_scores()
