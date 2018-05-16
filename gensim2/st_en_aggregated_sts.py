"""
@Project   : DuReader
@Module    : st_en_aggregated_sts.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/16/18 2:58 PM
@Desc      : 
"""

import os
import re
import string
from io import StringIO

import numpy as np
import pandas
import scipy.stats
import tensorflow as tf
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence


def load_sts_dataset(filename):
    # Loads a subset of the STS dataset into a DataFrame. In particular both
    # sentences and their human rated similarity score.
    sent_pairs = []
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            # (sent_1, sent_2, similarity_score)
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pandas.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def download_and_load_sts_data():
    sts_dataset = tf.keras.utils.get_file(
      fname="Stsbenchmark.tar.gz",
      origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
      extract=True)

    print('Dirname of sts_dataset:', os.path.dirname(sts_dataset))
    # location of the downloaded files

    sts_dev = load_sts_dataset(
      os.path.join(os.path.dirname(sts_dataset),
                   "stsbenchmark", "sts-dev.csv"))
    sts_test = load_sts_dataset(
      os.path.join(
          os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))

    return sts_dev, sts_test


def clean_sentence(st):
    """
    数据预处理
    :param st: string
    :return: string
    """
    in_tab = '[' + string.punctuation + '。，“”‘’（）：；？·—《》、' + ']'
    out_tab = ''
    clean = re.sub(in_tab, out_tab, st)
    return clean


def file_generate(sts):
    # seg_list = [' '.join(jieba.cut(st)) for st in sts]
    seg_list = [clean_sentence(st) for st in sts]
    handle = StringIO('\n'.join(seg_list))
    return handle


def model_load():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(file_dir,
                         'data/GoogleNews-vectors-negative300.bin.gz')
    model0 = KeyedVectors.load_word2vec_format(fname, binary=True)
    print('The model was loaded.')
    return model0


model = model_load()
vocab_dict = model.wv.vocab


def cal_features(messages):
    st_vector_list = []

    print('messages:')
    for message in LineSentence(file_generate(messages)):
        print(message)
        # LineSentence并没有把标点符号给去掉
        words_in_model = [word for word in message if word in vocab_dict]
        print(words_in_model)
        word_vectors = [model.wv[word].tolist() for word in words_in_model]
        st_matrix = np.array(word_vectors)
        # print(np.mean(st_matrix, axis=0).shape)
        st_vector = np.mean(st_matrix, axis=0).tolist()
        # print('Length of sentence vector:', len(st_vector))
        st_vector = st_vector/np.linalg.norm(st_vector)
        st_vector_list.append(st_vector)

        print()

    return st_vector_list


def cal_similarity_score(sts_dev):

    # def run_sts_benchmark(sess):
    #     """Returns the similarity scores"""
    #     emba, embb, scores0 = sess.run(
    #       [sts_encode1, sts_encode2, sim_scores],
    #       feed_dict={
    #           sts_input1: text_a,
    #           sts_input2: text_b
    #       })
    #     return scores0

    text_a = sts_dev['sent_1'].tolist()
    text_b = sts_dev['sent_2'].tolist()

    # sts_input1 = tf.placeholder(tf.string, shape=(None,))
    # sts_input2 = tf.placeholder(tf.string, shape=(None,))

    embed = cal_features
    # cal_features已经做了normalize

    # sts_encode1 = tf.nn.l2_normalize(embed(sts_input1))
    # sts_encode2 = tf.nn.l2_normalize(embed(sts_input2))
    sts_encode1 = embed(text_a)
    sts_encode2 = embed(text_b)

    # sim_scores = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
    sim_scores = np.sum(np.multiply(sts_encode1, sts_encode2), axis=1)
    sim_scores = sim_scores.tolist()

    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())
    #     session.run(tf.tables_initializer())
    #     scores = run_sts_benchmark(session)

    return sim_scores


def cal_pearson_correlation(scores, dev_scores):

    pearson_correlation = scipy.stats.pearsonr(scores, dev_scores)
    # 两个分布的pearson分布系数，用直线拟合，但目标函数时点到直线距离最小
    print('Pearson correlation coefficient = {0}\np-value = {1}'.format(
        pearson_correlation[0], pearson_correlation[1]))


if __name__ == '__main__':

    sts_dev0, sts_test0 = download_and_load_sts_data()

    scores1 = cal_similarity_score(sts_dev0)

    dev_scores0 = sts_dev0['sim'].tolist()

    print(len(scores1), len(dev_scores0))

    cal_pearson_correlation(scores1, dev_scores0)

    # 1500 1500
    # Pearson correlation coefficient = 0.697644100820568
    # p - value = 3.416875480477643e-219
