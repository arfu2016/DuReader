"""
@Project   : DuReader
@Module    : one_vs_one.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/13/18 11:51 AM
@Desc      : 两两成对的句子相似度的计算
"""

import os

import pandas
import scipy.stats
import tensorflow as tf
import tensorflow_hub as hub


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

    def run_sts_benchmark(sess):
        """Returns the similarity scores"""
        emba, embb, scores0 = sess.run(
          [sts_encode1, sts_encode2, sim_scores],
          feed_dict={
              sts_input1: text_a,
              sts_input2: text_b
          })
        return scores0

    text_a = sts_dev['sent_1'].tolist()
    # pd.series to list
    text_b = sts_dev['sent_2'].tolist()

    sts_input1 = tf.placeholder(tf.string, shape=(None,))
    # 一维向量，元素个数不限；矩阵处理可以并行，速度上胜过for循环
    sts_input2 = tf.placeholder(tf.string, shape=(None,))

    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)))
    embed = hub.Module(os.path.join(base_dir,
                                    'tf-hub2/data/universal-sentence-encoder'))

    # For evaluation we use exactly normalized rather than
    # approximately normalized.
    sts_encode1 = tf.nn.l2_normalize(embed(sts_input1))
    # By l2_normalize we get the norm of the vector, 得到的是需要feed_dict的tensor
    # 是一个矩阵
    sts_encode2 = tf.nn.l2_normalize(embed(sts_input2))

    sim_scores = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        scores = run_sts_benchmark(session)

    return scores


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
