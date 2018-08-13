"""
@Project   : DuReader
@Module    : download_sts_data.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/13/18 11:05 AM
@Desc      : 下载sts数据集
"""
import os

import pandas as pd
import tensorflow as tf
from work2.logger_setup import define_logger
logger = define_logger('work2.download_sts_data')


def load_sts_dataset(filename):
    # Loads a subset of the STS dataset into a DataFrame. In particular both
    # sentences and their human rated similarity score.
    sent_pairs = []
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def download_and_load_sts_data():
    sts_dataset = tf.keras.utils.get_file(
      fname="Stsbenchmark.tar.gz",
      origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
      extract=True)

    logger.debug(
        'Dirname of sts_dataset: {}'.format(os.path.dirname(sts_dataset)))
    # location of the downloaded files

    sts_dev = load_sts_dataset(
      os.path.join(os.path.dirname(sts_dataset),
                   "stsbenchmark", "sts-dev.csv"))
    sts_test = load_sts_dataset(
      os.path.join(
          os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))

    return sts_dev, sts_test


def get_and_test():
    sts_dev, sts_test = download_and_load_sts_data()
    print('Test dataset:')
    print(sts_test.to_string())


if __name__ == '__main__':
    get_and_test()
