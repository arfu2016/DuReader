"""
@Project   : DuReader
@Module    : pearson.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/17/18 10:25 AM
@Desc      : 
"""
import os

import numpy as np
import pandas
import scipy.stats
import tensorflow as tf


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


def cal_pearson_correlation(scores, dev_scores):

    pearson_correlation = scipy.stats.pearsonr(scores, dev_scores)
    # 两个分布的pearson分布系数，用直线拟合，但目标函数时点到直线距离最小
    print('Pearson correlation coefficient = {0}\np-value = {1}'.format(
        pearson_correlation[0], pearson_correlation[1]))


if __name__ == '__main__':

    sts_dev0, sts_test0 = download_and_load_sts_data()

    dev_scores0 = sts_dev0['sim'].tolist()

    scores1 = np.random.uniform(low=0, high=5, size=len(dev_scores0))

    print(len(scores1), len(dev_scores0))

    cal_pearson_correlation(scores1, dev_scores0)
    # Pearson correlation coefficient = 0.02144320357998576
    # p - value = 0.406598277363874
