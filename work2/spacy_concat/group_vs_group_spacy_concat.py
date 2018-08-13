"""
@Project   : DuReader
@Module    : group_vs_group_spacy_concat.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/13/18 3:43 PM
@Desc      : 两组句子相似度的计算，暂时要求每组的句子数目是相同的
"""
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spacy
from cachetools import cached, TTLCache

from work2.logger_setup import define_logger

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)))
cache = TTLCache(maxsize=100, ttl=300)
logger = define_logger('work2.group_vs_group_spacy_concat')


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


@cached(cache)
def _single_sentence(sentence: str) -> list:
    word_list = nlp(sentence)
    print('word_list:', [word.text for word in word_list])
    word_list = [word for word in word_list if word.text not in punc_set]
    # 去除句中的标点符号
    word_vectors = [word.vector for word in word_list
                    if word.has_vector]
    print('Effective words:', [word.text for word in word_list
                               if word.has_vector])
    st_vector = concat_pooling(word_vectors)

    return st_vector


@cached(cache)
def get_features(sentences: tuple) -> np.ndarray:
    """use sentences as a tuple is to be consistent with tf.hub"""
    sen_embedding = [_single_sentence(st) for st in sentences]
    sen_embedding = np.array(sen_embedding)
    return sen_embedding


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


if __name__ == '__main__':

    messages = (
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

    embeddings = get_features(messages)

    plot_similarity(messages, embeddings, 90)
