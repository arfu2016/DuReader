"""
@Project   : DuReader
@Module    : one_vs_group_google.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/13/18 12:02 PM
@Desc      : 给出一组句子，找到其中和一个另给的句子最相似的句子
"""
import os
import pprint
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from cachetools import cached, TTLCache

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)))
embed = hub.Module(os.path.join(base_dir,
                                'tf-hub2/data/universal-sentence-encoder'))
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
    # 计算点积

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

    with ProcessPoolExecutor(2) as executor:
        sim_scores = executor.map(VectorSimilarity(test_vector),
                                  training_vectors)
    # the executor.__exit__ method will call executor.shutdown(wait=True) ,
    # which will block until all processes are done.
    # 也就是说，executor.map()是阻塞调用，返回的是一个iterator
    # 此时类似函数式编程，返回的是函数表达式

    return list(sim_scores)
    # 展成list的过程就是函数表达式计算的过程，也可以用for或者next()来展开计算


# 给定一个句子和一组句子，找出后者各个句子中和前者最为相似的句子

def most_similar(training_sentences: tuple, test_sentence: tuple) -> str:
    training_embeddings = _sentence_embedding(training_sentences)
    start = time.perf_counter()
    test_embedding = _sentence_embedding(test_sentence)
    print(
        'time to get sentence vector: {:.3f}'.format(
            time.perf_counter()-start))
    sim_scores = _similarity_scores(training_embeddings, test_embedding)
    print('sim_scores:')
    formatted_sim_scores = [float('{:.3f}'.format(value))
                            for value in sim_scores]
    pprint.pprint(formatted_sim_scores)
    idx = np.argmax(sim_scores)
    # 此处要求sim_scores是list，如果只是iterator，很可能出错
    return training_sentences[idx]


# test

def t_most_similar():
    start = time.perf_counter()
    print()
    print('Test t_most_similar()')
    training_sentences = (
        # Smartphones
        "I like my phone",
        "My phone is not good.",
        "Your cellphone looks great.",

        # Weather
        "Will it snow tomorrow?",
        "Recently a lot of hurricanes have hit the US",
        "Global warming is real",

        # Food and health
        "An apple a day, keeps the doctors away",
        "Eating strawberries is healthy",
        "Is paleo better than keto?",

        # Asking about age
        "How old are you?",
        "what is your age?",
    )
    test_sentence = ('How is the weather today?',)
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'How is the weather today?':")
    print(top_sentence)
    print('time of t_most_similar(): {:.3f}'.format(
        time.perf_counter() - start))

    session.close()


if __name__ == '__main__':

    t_most_similar()
