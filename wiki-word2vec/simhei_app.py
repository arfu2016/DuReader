"""
@Project   : DuReader
@Module    : simhei_app.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/9/18 11:16 AM
@Desc      : 
"""
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontManager


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
