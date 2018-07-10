"""
@Project   : DuReader
@Module    : sen_sim_w2v_ngram.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/10/18 10:03 AM
@Desc      : 
"""
import logging
import os
import string
import time
# from multiprocessing import Pool

import numpy as np
import spacy
from cachetools import cached, TTLCache

file_dir = os.path.dirname(os.path.dirname(__file__))
cache = TTLCache(maxsize=100, ttl=300)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


# 把句子转换为向量

def model_load():
    nlp0 = spacy.load('en_core_web_md')
    logging.info('The model was loaded.')
    return nlp0


nlp = model_load()


def init_punctuation():
    punc_string = string.punctuation + '。，“”‘’（）：；？·—《》、'
    punc_set0 = {punc for punc in punc_string}
    return punc_set0


punc_set = init_punctuation()


def avg_pooling(word_vectors: list) -> list:
    st_matrix = np.array(word_vectors)
    st_vector = np.mean(st_matrix, axis=0)
    st_vector = st_vector / np.linalg.norm(st_vector)
    st_vector = st_vector.tolist()
    return st_vector


def max_pooling(word_vectors: list) -> list:
    st_matrix = np.array(word_vectors)
    st_vector = np.max(st_matrix, axis=0)
    st_vector = st_vector / np.linalg.norm(st_vector)
    st_vector = st_vector.tolist()
    return st_vector


def concat_pooling(word_vectors: list) -> list:
    st_vector1 = avg_pooling(word_vectors)
    st_vector2 = max_pooling(word_vectors)
    st_vector1.extend(st_vector2)
    return st_vector1


def ngram_pooling(word_vectors: list, n: int =2) -> list:
    if len(word_vectors) >= n:
        st_vectors = []
        for i in range(len(word_vectors)-n+1):
            st_matrix0 = np.array(word_vectors[i: i+n])
            st_vector0 = np.mean(st_matrix0, axis=0)
            st_vectors.append(st_vector0)

        st_matrix = np.array(st_vectors)
        st_vector = np.max(st_matrix, axis=0)
        st_vector = st_vector / np.linalg.norm(st_vector)
        st_vector = st_vector.tolist()
    else:
        st_vector = avg_pooling(word_vectors)
    return st_vector


@cached(cache)
def _single_sentence(sentence: str) -> list:
    word_list = nlp(sentence)
    word_list = [word for word in word_list if word.text not in punc_set]
    word_vectors = [word.vector for word in word_list
                    if word.has_vector]
    print('word_list:', word_list)
    print('Effective words:', [word for word in word_list if word.has_vector])
    # st_vector = avg_pooling(word_vectors)
    # st_vector = max_pooling(word_vectors)
    # st_vector = concat_pooling(word_vectors)
    st_vector = ngram_pooling(word_vectors)

    return st_vector


@cached(cache)
def _sentence_embedding(sentences: tuple) -> np.ndarray:
    """use sentences as a tuple is to be consistent with tf.hub"""
    sen_embedding = [_single_sentence(st) for st in sentences]
    # with Pool(2) as p:
    #     sen_embedding = p.map(_single_sentence, sentences)
    sen_embedding = np.array(sen_embedding)
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

    sim_scores = [_vector_similarity(vector, test_vector)
                  for vector in training_vectors]

    return sim_scores


# 给定一个句子和一组句子，找出后者各个句子中和前者最为相似的句子

def most_similar(training_sentences: tuple, test_sentence: str) -> str:
    training_embeddings = _sentence_embedding(training_sentences)
    test_embedding = _single_sentence(test_sentence)
    sim_scores = _similarity_scores(training_embeddings, test_embedding)
    print('sim_scores:', sim_scores)
    idx = np.argmax(sim_scores)
    return training_sentences[idx]


# test

def t_most_similar():
    start = time.perf_counter()
    print()
    print('Test t_most_similar()')
    training_sentences = ("The quick brown fox jumps over the lazy dog.",
                          "Who is Messy?")
    test_sentence = 'Something about football.'
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'Something about football.':")
    print(top_sentence)
    print('time of t_most_similar():', time.perf_counter() - start)


def t2_most_similar():
    start = time.perf_counter()
    print()
    print('Test t2_most_similar()')
    training_sentences = ("The quick brown fox jumps over the lazy dog.",
                          "Who is Messy?")
    test_sentence = 'Can you tell me something about Cristiano Ronaldo?'
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'Can you tell me something about Cristiano Ronaldo?':")
    print(top_sentence)
    print('time of t2_most_similar():', time.perf_counter() - start)


if __name__ == '__main__':

    t_most_similar()
    t2_most_similar()
