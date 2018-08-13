"""
@Project   : DuReader
@Module    : group_vs_group.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/13/18 11:45 AM
@Desc      : 两组句子相似度的计算，暂时要求每组的句子数目是相同的
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub


def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    # features是多个sentence embeddings组成的矩阵，上面这个内积操作就算出了sentence两两之间的相似度
    # np.inner(u, v) == u*transpose(v)
    sns.set(font_scale=1.2)
    g = sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        # 如果值小于0，就按0处理，也就是所谓的语境相反会被当做不相关处理
        vmax=1,
        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    # 把labels字体进行旋转
    g.set_title("Semantic Textual Similarity")
    plt.show()


def get_features(labels):
    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)))
    embed = hub.Module(os.path.join(base_dir,
                                    'tf-hub2/data/universal-sentence-encoder'))
    similarity_input_placeholder = tf.placeholder(tf.string, shape=(None,))
    # None表示可以是任意值，此处只给出维度，是一维向量，至于有几个元素，没有限制
    similarity_message_encodings = embed(similarity_input_placeholder)
    # 此处传给embed operation的不再是常数，而是一个placeholder，得到tensor
    # 该tensor中包含placeholder系数，在session.run的时候需要提供feed_dict，key是placeholder系数的名字
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        features = session.run(similarity_message_encodings,
                               feed_dict={similarity_input_placeholder: labels}
                               )
    return features


if __name__ == '__main__':

    messages = [
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
    ]

    embeddings = get_features(messages)

    plot_similarity(messages, embeddings, 90)
