"""
@Project   : DuReader
@Module    : one_vs_group.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/15/18 1:39 PM
@Desc      : 给出一组句子，找到其中和一个另给的句子最相似的句子
"""
import os
import pprint
import string
import time
from concurrent.futures import ProcessPoolExecutor
from io import StringIO

import jieba
import numpy as np
import gensim
from cachetools import cached, TTLCache
from gensim.models.word2vec import LineSentence

from work2.logger_setup import define_logger

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
cache = TTLCache(maxsize=100, ttl=300)
logger = define_logger('work2.one_vs_group_spacy_concat')


# 把句子转换为向量

def model_load():
    fn = os.path.join(base_dir, "wiki-word2vec/data/wiki.zh.model")
    model0 = gensim.models.Word2Vec.load(fn)
    print('The model was loaded.')
    return model0


def avg_pooling(word_vectors: list) -> list:
    st_matrix = np.array(word_vectors)
    st_vector = np.mean(st_matrix, axis=0)
    st_vector = st_vector
    st_vector = st_vector.tolist()
    return st_vector


def max_pooling(word_vectors: list) -> list:
    st_matrix = np.array(word_vectors)
    st_vector = np.max(st_matrix, axis=0)
    st_vector = st_vector
    st_vector = st_vector.tolist()
    return st_vector


def concat_pooling(word_vectors: list) -> list:
    st_vector1 = avg_pooling(word_vectors)
    st_vector2 = max_pooling(word_vectors)
    st_vector1.extend(st_vector2)
    st_vector1 = st_vector1 / np.linalg.norm(st_vector1)
    return st_vector1


# @cached(cache)
# def _single_sentence(sentence: str) -> list:
#     word_list = nlp(sentence)
#     word_list = [word for word in word_list if word.text not in punc_set]
#     # 去除句中的标点符号
#     word_vectors = [word.vector for word in word_list
#                     if word.has_vector]
#     st_vector = concat_pooling(word_vectors)
#
#     return st_vector

def file_generate(sts):
    seg_list = [' '.join(jieba.cut(st)) for st in sts]
    handle = StringIO('\n'.join(seg_list))
    return handle


@cached(cache)
def _sentence_embedding(sentences: tuple) -> np.ndarray:
    model = model_load()
    vocab_dict = model.wv.vocab
    # 是用去掉标点后的wiki训练的，所以vocab中没有标点符号

    st_vector_list = []

    print('messages:')
    for message in LineSentence(file_generate(sentences)):
        print(message)
        words_in_model = [word for word in message if word in vocab_dict]
        print(words_in_model)
        word_vectors = [model.wv[word].tolist() for word in words_in_model]
        # st_matrix = np.array(word_vectors)
        # st_vector = np.mean(st_matrix, axis=0).tolist()
        # st_vector = st_vector/np.linalg.norm(st_vector)
        st_vector = concat_pooling(word_vectors)
        st_vector_list.append(st_vector)
        print()
    return np.array(st_vector_list)


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

def most_similar(training_sentences: tuple, test_sentence: tuple) -> list:
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
    sentence_score = zip(training_sentences, sim_scores)
    sentence_score = sorted(sentence_score, key=lambda x: x[1], reverse=True)
    # sentence_score[0:5]
    sentence_score = [
        (sentence, score) for sentence, score in sentence_score
        if score > 0.8]
    return sentence_score


# test

def t_most_similar():
    start = time.perf_counter()
    print()
    print('Test t_most_similar()')
    training_sentences = (
        # 不在顶级联赛
        "哪些球员球队如今不在顶级联赛?",
        "现在哪些球员所在的球队不在顶级联赛了",
        "有没有什么球员现在不在顶级联赛踢球了",

        # 死敌
        "哪些球员的俱乐部是死敌",
        "现在哪些球员所在球队是死敌",
        "哪些球员所在的俱乐部势不两立",

        # 效力
        "梅西效力于哪家俱乐部",
        "梅西在哪个球队踢球",
        "梅西在哪个俱乐部踢球",

        # Asking about age
        "你多大了",
        "你的年龄是多少",
    )
    test_sentence = ('罗纳尔多在哪里工作?',)
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'How is the weather today?':")
    print(top_sentence)
    print('time of t_most_similar(): {:.3f}'.format(
        time.perf_counter() - start))


if __name__ == '__main__':

    t_most_similar()
