"""
@Project   : DuReader
@Module    : st_vector_kmeans_comp.py
@Author    : Deco [deco@cubee.com]
@Created   : 6/22/18 1:45 PM
@Desc      : 
"""
import os
import sys
import pprint

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(__file__)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import st_vector_produce as svp

iris = load_iris()


def sklearn_kMeans(data):
    model = KMeans(n_clusters=4, random_state=0)
    model.fit(data)
    # print('centers:', model.cluster_centers_)
    # print('labels:', model.labels_)
    # print('distances:', model.transform(data))
    distances = model.transform(data)
    return model.labels_, distances


def points_select(label, vectors, labels):

    index0 = np.nonzero(labels == label)[0].tolist()
    x0 = vectors
    k0 = 2
    x, y = svp.cal_scatter(svp.sk_pca, x0, k0)
    point0 = [xy for idx, xy in enumerate(zip(x, y))
              if idx in index0]
    x0, y0 = zip(*point0)
    return x0, y0


def select_sentences(distances):
    n_clusters = 4
    num_each_cluster = 2
    distances = distances.tolist()
    top_select = []
    for i in range(n_clusters):
        distance_vector = [(ele[i], idx) for idx, ele in enumerate(distances)]
        distance_vector.sort(key=lambda x: x[0])
        top_select.append(distance_vector[:num_each_cluster])
    return top_select


def plot_pca(vectors, i):
    labels, distances = sklearn_kMeans(vectors)
    top_select = select_sentences(distances)
    x0, y0 = points_select(0, vectors, labels)
    x1, y1 = points_select(1, vectors, labels)
    x2, y2 = points_select(2, vectors, labels)
    x3, y3 = points_select(3, vectors, labels)

    fig = plt.figure(i)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x0, y0, color='blue')
    ax.scatter(x1, y1, color='red')
    ax.scatter(x2, y2, color='green')
    ax.scatter(x3, y3, color='black')

    return top_select


if __name__ == '__main__':

    sentences = svp.load_es0620()
    embeddings = svp.cal_features(sentences)
    select_idx = plot_pca(embeddings, 1)
    pprint.pprint(select_idx)

    for ele in select_idx:
        first = ele[0][1]
        second = ele[1][1]
        print()
        print(sentences[first])
        print(sentences[second])

    sentences = svp.load_es0622()
    embeddings = svp.cal_features(sentences)
    select_idx = plot_pca(embeddings, 2)
    pprint.pprint(select_idx)

    for ele in select_idx:
        first = ele[0][1]
        second = ele[1][1]
        print()
        print(sentences[first])
        print(sentences[second])

    plt.show()
