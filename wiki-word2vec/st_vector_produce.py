"""
@Project   : DuReader
@Module    : st_vector_produce.py
@Author    : Deco [deco@cubee.com]
@Created   : 6/19/18 4:28 PM
@Desc      : 
"""
from io import StringIO
# import pprint

import elasticsearch
import gensim
import jieba
import matplotlib.pyplot as plt
import numpy as np
from gensim.models.word2vec import LineSentence
from sklearn import decomposition

es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])


def model_load():
    model0 = gensim.models.Word2Vec.load("data/wiki.zh.model")
    print('The model was loaded.')
    return model0


def file_generate(sts):
    seg_list = [' '.join(jieba.cut(st)) for st in sts]
    handle = StringIO('\n'.join(seg_list))
    return handle


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
        st_vector = np.mean(st_matrix, axis=0).tolist()
        st_vector = st_vector/np.linalg.norm(st_vector)
        st_vector_list.append(st_vector)
        print()

    return st_vector_list


def search_data():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={"_source": ["title", "description"],
                        "query": {"match_all": {}},
                        "size": 1000})
    # pprint.pprint(p)
    return p


def load_es():
    news_dict = search_data()
    news_list = news_dict['hits']['hits']
    st = [news['_source']['title'][0] for news in news_list]
    # pprint.pprint(st)
    return st


def search_data0620():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={"_source": ["title", "description"],
                        "query": {"bool": {"filter": {
                            "range": {"display_time": {
                                "gte": '2018-06-20 00:00:00',
                                "lt": '2018-06-21 00:00:00',
                                }},
                            }}},
                        "size": 1000})
    # pprint.pprint(p)
    return p


def load_es0620():
    news_dict = search_data0620()
    news_list = news_dict['hits']['hits']
    st = [news['_source']['title'][0] for news in news_list]
    # pprint.pprint(st)
    return st


def search_data0622():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={"_source": ["title", "description"],
                        "query": {"bool": {"filter": {
                            "range": {"display_time": {
                                "gte": '2018-06-22 00:00:00',
                                "lt": '2018-06-23 00:00:00',
                                }},
                            }}},
                        "size": 1000})
    # pprint.pprint(p)
    return p


def load_es0622():
    news_dict = search_data0622()
    news_list = news_dict['hits']['hits']
    st = [news['_source']['title'][0] for news in news_list]
    # pprint.pprint(st)
    return st


def sk_pca(X, k):
    """
    PCA can be implemented either by finding the eigenvectors of the covariance
    matrix or by applying SVD to the centered data matrix. You don't do both
    covariance and SVD. In practice, the SVD approach is preferable because
    the computation of covariance matrix amplifies numerical issues associated
    with poorly conditioned matrices.
    sklearn uses the SVD method, and applies data centering (remove mean) in the
    program

    https://stackoverflow.com/questions/47476209/why-does-sklearn-decomposition-pca-fit-transformx-no-multiplication-by-x
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/pca.py#L432
    :param X:
    :param k:
    :return:
    """
    pca2 = decomposition.PCA(n_components=k, svd_solver='full')
    pca2.fit(X)
    # adjust_components = pca2.components_ + pca2.mean_
    X_reduced = pca2.transform(X)

    return X_reduced


def cal_scatter(func, X, k):
    X_pca = func(X, k)
    X_pca = X_pca.tolist()
    xy = zip(*X_pca)
    return xy


def plot_hist(ele):
    plt.hist(ele, bins='auto')
    # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()


def plot_pca(vectors):
    x0 = vectors
    k0 = 2
    x2y2 = cal_scatter(sk_pca, x0, k0)
    x2, y2 = x2y2
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x2, y2, color='blue')
    plt.show()


if __name__ == '__main__':

    sentences = load_es()

    embeddings = cal_features(sentences)
    # print('embeddings:')
    # pprint.pprint(embeddings)

    first_ele = [emb[1] for emb in embeddings]
    positive_ele = [ele for ele in first_ele if ele > 0]
    print('min and max positive numbers:',
          (min(positive_ele), max(positive_ele)))
    plot_hist(first_ele)

    plot_pca(embeddings)
