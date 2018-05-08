"""
@Project   : DuReader
@Module    : st_fasttext_aggregated.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/8/18 6:26 PM
@Desc      : 
"""

import subprocess
from io import StringIO
import os

import jieba
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gensim.models.word2vec import LineSentence
from matplotlib.font_manager import FontManager
from gensim.models import KeyedVectors


def plot_similarity(labels, features, rotation):
    """
    sentence vector evaluation
    :param labels: list of sentences
    :param features: matrix composed by sentence vectors
    :param rotation: generally 90
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['font.serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    print('matplotlib directory:', mpl.matplotlib_fname())
    # SimHei.ttf should be in
    # /home/deco/miniconda2/envs/tf17/lib/python3.6/site-packages/matplotlib/
    # mpl-data/fonts/ttf

    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist)
    output = subprocess.check_output(
        'fc-list :lang=zh -f "%{family}\n"', shell=True)
    # print('Type of output:', type(output))
    output = output.decode('utf-8')
    # print('*' * 10, '系统可用的中文字体', '*' * 10)
    # print(output)
    zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
    available = mat_fonts & zh_fonts
    print('*' * 10, '可用的字体', '*' * 10)
    for f in available:
        print(f)

    corr = np.inner(features, features)
    # features是多个sentence embeddings组成的矩阵，上面这个内积操作就算出了sentence两两之间的相似度
    # np.inner(u, v) == u*transpose(v)
    sns.set(font_scale=1.2)
    sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Arial']})
    # the code above must be after sns.set()
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


def file_generate(sts):
    seg_list = [' '.join(jieba.cut(st)) for st in sts]
    handle = StringIO('\n'.join(seg_list))
    return handle


def model_load():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(file_dir, 'data/wiki.zh.vec')
    model0 = KeyedVectors.load_word2vec_format(fname, binary=False)
    print('The model was loaded.')
    return model0


def cal_features(messages):
    model = model_load()
    vocab_dict = model.wv.vocab

    st_vector_list = []

    print('messages:')
    for message in LineSentence(file_generate(messages)):
        print(message)
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


if __name__ == '__main__':

    sentences = [
        # 不在顶级联赛
        "哪些球员球队如今不在顶级联赛",
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
    ]

    embeddings = cal_features(sentences)

    plot_similarity(sentences, embeddings, 90)
