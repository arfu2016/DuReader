"""
@Project   : DuReader
@Module    : download_sts_benchmark.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/3/18 4:58 PM

@Desc      :
sts benchmark to evaluate methods for sentence similarity calculation
https://colab.research.google.com/github/tensorflow/hub/blob/master/examples
/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb
#scrollTo=VOs8ZfOnJeBF
"""

import pandas
import os
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


sts_dev0, sts_test0 = download_and_load_sts_data()

file_dir = os.path.dirname(__file__)
sts_dev1 = load_sts_dataset(
    os.path.join(file_dir, "sts-bench-mark",
                 "stsbenchmark", "sts-dev.csv"))
sts_test1 = load_sts_dataset(
    os.path.join(file_dir, "sts-bench-mark",
                 "stsbenchmark", "sts-test.csv"))
