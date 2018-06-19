"""
@Project   : DuReader
@Module    : st_vector_produce.py
@Author    : Deco [deco@cubee.com]
@Created   : 6/19/18 4:28 PM
@Desc      : 
"""
import subprocess
from io import StringIO

import gensim
import jieba
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gensim.models.word2vec import LineSentence
from matplotlib.font_manager import FontManager
import pprint
import elasticsearch

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
        # print(np.mean(st_matrix, axis=0).shape)
        st_vector = np.mean(st_matrix, axis=0).tolist()
        # print('Length of sentence vector:', len(st_vector))
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


if __name__ == '__main__':

    sentences = load_es()

    embeddings = cal_features(sentences)
    # print('embeddings:')
    # pprint.pprint(embeddings)
