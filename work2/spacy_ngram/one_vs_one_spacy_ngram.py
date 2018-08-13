"""
@Project   : DuReader
@Module    : one_vs_one_spacy_ngram.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/13/18 3:37 PM
@Desc      : 两两成对的句子相似度的计算
"""

import pandas
import scipy.stats
import tensorflow as tf

import os
import string

import numpy as np
import spacy
from cachetools import cached, TTLCache

from work2.logger_setup import define_logger

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)))
cache = TTLCache(maxsize=100, ttl=300)
logger = define_logger('work2.one_vs_one_spacy_ngram')


# 把句子转换为向量

def model_load():
    nlp_spacy = spacy.load('en_core_web_md')
    logger.info('The model was loaded.')
    return nlp_spacy


nlp = model_load()


def init_punctuation():
    punc_string = string.punctuation + '。，“”‘’（）：；？·—《》、'
    punc_set_english_chinese = {punc for punc in punc_string}
    return punc_set_english_chinese


punc_set = init_punctuation()


def avg_pooling(word_vectors: list) -> list:
    st_matrix = np.array(word_vectors)
    st_vector = np.mean(st_matrix, axis=0)
    st_vector = st_vector / np.linalg.norm(st_vector)
    st_vector = st_vector.tolist()
    return st_vector


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
    # 去除句中的标点符号
    word_vectors = [word.vector for word in word_list
                    if word.has_vector]
    st_vector = ngram_pooling(word_vectors, n=2)

    return st_vector


# 给定两个句向量，计算向量相似度

def _vector_similarity(encode1: list, encode2: list) -> float:
    """assume the length of encode1 and encode2 are n, time complexity is
    O(n), space complexity is O(n)
    """
    sim_score = sum([x*y for x, y in zip(encode1, encode2)])

    return sim_score


def load_sts_dataset(filename):
    # Loads a subset of the STS dataset into a DataFrame. In particular both
    # sentences and their human rated similarity score.
    sent_pairs = []
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pandas.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def download_and_load_sts_data():
    sts_dataset = tf.keras.utils.get_file(
      fname="Stsbenchmark.tar.gz",
      origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
      extract=True)

    sts_dev = load_sts_dataset(
      os.path.join(os.path.dirname(sts_dataset),
                   "stsbenchmark", "sts-dev.csv"))
    sts_test = load_sts_dataset(
      os.path.join(
          os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))

    return sts_dev, sts_test


def cal_similarity_score(sts_dev):

    text_a = sts_dev['sent_1'].tolist()
    # pd.series to list
    text_b = sts_dev['sent_2'].tolist()

    sim_scores = []
    for text1, text2 in zip(text_a, text_b):
        encode1 = _single_sentence(text1)
        encode2 = _single_sentence(text2)
        sim_score = _vector_similarity(encode1, encode2)
        sim_scores.append(sim_score)

    return sim_scores


def cal_pearson_correlation(scores, dev_scores):

    pearson_correlation = scipy.stats.pearsonr(scores, dev_scores)
    # 两个分布的pearson分布系数，用直线拟合，但目标函数时点到直线距离最小
    print('Pearson correlation coefficient = {0}\np-value = {1}'.format(
        pearson_correlation[0], pearson_correlation[1]))


def get_and_test():
    sts_dev0, sts_test0 = download_and_load_sts_data()
    scores0 = cal_similarity_score(sts_dev0)
    dev_scores0 = sts_dev0['sim'].tolist()
    cal_pearson_correlation(scores0, dev_scores0)


if __name__ == '__main__':
    get_and_test()
