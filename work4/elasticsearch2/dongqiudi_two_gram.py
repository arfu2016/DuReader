"""
@Project   : text-classification-cnn-rnn
@Module    : dongqiudi_two_gram.py
@Author    : Deco [deco@cubee.com]
@Created   : 6/22/18 2:08 PM
@Desc      : 
"""
import json
import logging
import os
import pprint
import string
import sys
import re

import elasticsearch
import jieba
import requests
from elasticsearch.helpers import bulk
from cachetools import cached, TTLCache

cache = TTLCache(maxsize=100, ttl=300)

es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])


def set_logger():
    # for logging

    log_path = 'logs/dongqiudi.txt'  # ''

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def create_mapping():
    setting = {
        "mappings": {
            "news": {
                "properties": {
                    "date": {
                        "type": "date",
                        "format": "YYYY-MM-DD HH:mm:ss"
                    },
                    "display_time": {
                        "type": "date",
                        "format": "YYYY-MM-DD HH:mm:ss"
                    }
                }
            }
        }
    }

    es.indices.create(index='dongqiudi', body=setting)


def put_data(tpl_intent_entity):
    i = 1
    for tie in tpl_intent_entity:
        es.index(index='dongqiudi', doc_type='news', id=i,
                 body=tie)
        if i % 200 == 0:
            print(i)
        i = i + 1


def set_data(list_data_dict, index_name="dongqiudi", doc_type_name="news"):
    """https://github.com/elastic/elasticsearch-py/issues/508"""

    unique_set = set()

    for data_dict in list_data_dict:
        if data_dict['title'][0] not in unique_set:
            unique_set.add(data_dict['title'][0])
            yield {
                "_index": index_name,
                "_type": doc_type_name,
                "_source": data_dict
            }

    # for idx, data_dict in enumerate(list_data_dict):
    #     yield {
    #         "_index": index_name,
    #         "_type": doc_type_name,
    #         "_id": idx,
    #         "_source": data_dict
    #     }


def bulk_put(tpl_intent_entity, **kwargs):
    success, _ = bulk(es, set_data(tpl_intent_entity, **kwargs))
    print("insert %s lines" % success)


def dict2es():
    file_dir = os.path.dirname(os.path.dirname(__file__))
    fn = os.path.join(file_dir, 'scrapy2/dongqiudi/items.json')
    with open(fn, 'r', encoding='utf-8') as f:
        list_news_dict = json.load(f)

    bulk_put(list_news_dict)


def construct_query(list_data_dict):

    search_arr = []

    for data_dict in list_data_dict:
        # req_head
        search_arr.append({'index': 'dongqiudi', 'type': 'news'})
        # req_body
        search_arr.append({"query": {"bool": {"filter": {"bool": {"must": [
                      {"term": {"title.keyword": data_dict['title'][0]}},
                  ]}}}}})

    request = ''
    for each in search_arr:
        request += '%s \n' % json.dumps(each)

    return request


def check_existence(list_data_dict):
    res = es.msearch(body=construct_query(list_data_dict))
    # pprint.pprint(res)
    return [item['hits']['total'] for item in res['responses']]


def append_data():
    file_dir = os.path.dirname(os.path.dirname(__file__))
    fn = os.path.join(file_dir, 'scrapy2/dongqiudi/items.json')
    with open(fn, 'r', encoding='utf-8') as f:
        list_news_dict = json.load(f)

    tags = check_existence(list_news_dict)
    list_news_dict = [news_dict for idx, news_dict in enumerate(list_news_dict)
                      if tags[idx] == 0]
    # print('list_news_dict in test_check_existence():')
    # pprint.pprint(list_news_dict)

    bulk_put(list_news_dict)


def search_data_size():
    res = requests.get(
        'http://localhost:9200/dongqiudi/_search?size=5&from=10&pretty')
    pprint.pprint(json.loads(res.content))


def search_data_match_all0():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={"_source": ["title", "description"],
                        "query": {"match_all": {}},
                        "size": 1000})
    # pprint.pprint(p)
    return p


def search_data_match_all():
    test = es.search(index=["dongqiudi"], doc_type=["news"])
    size = test['hits']['total']
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={"_source": ["title", "description"],
                        "query": {"match_all": {}},
                        "size": size})
    # pprint.pprint(p)
    return p


def search_data_match():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={"_source": ["title", "description", "display_time"],
                        "query": {"bool": {
                         "must": {"match": {"title": "世界杯"}},
                        }}})
    pprint.pprint(p)


def search_data_term():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={
                        "query": {"bool": {"filter": {"bool": {
                         "must": {"term": {"title.keyword":
                          "7分钟透析俄罗斯世界杯的“中国队” | B面世界杯"}},
                        }}}}})
    pprint.pprint(p)


def search_data_date():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={
                        "query": {"constant_score": {"filter": {
                            "range": {"display_time": {
                                "gte": '2018-06-20 00:00:00',
                                }},
                            }}}})
    pprint.pprint(p)


def search_data_date2():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={
                        "query": {"bool": {"filter": {
                            "range": {"display_time": {
                                    "gte": '2018-06-20',
                                    "format": "YYYY-MM-DD"
                                    # "format": "yyyy-MM-dd"
                                    }}
                            }}}})
    pprint.pprint(p)


def search_data_date3():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={
                        "query": {"bool": {"filter": {
                            "range": {"display_time": {
                                    "gte": '2018-06-20 00:00:00',
                                    "lte": '2018-06-21 00:00:00',
                                    }}
                            }}}})
    pprint.pprint(p)


def count_data():
    logger = set_logger()
    r = requests.get('http://localhost:9200/dongqiudi/_count?pretty')
    info = json.loads(r.content)
    pprint.pprint(info)
    logger.info(
        'There are {} records in dongqiudi index'.format(info['count']))


def del_dongqiudi():
    es.indices.delete(index='dongqiudi', ignore=[400, 404])


def clean_sentence(st):
    in_tab = string.punctuation + '。，“”‘’（）：；？·—《》、'
    pt = set(p for p in in_tab)
    clean = ''.join([c for c in st if c not in pt])
    # hash search, time complexity m*O(1)
    return clean


@cached(cache)
def construct_stop_words():
    in_tab = string.punctuation + '。，“”‘’（）：；？！·—《》、'
    pt = set(p for p in in_tab)
    stop_words = {'全场', '出线', '球帝', '图集', '代表', '世界', '足球'}
    stop_words.update({'的', })
    stop_words.update(pt)
    return stop_words


def sentence2tags(st):
    stop_words = construct_stop_words()
    tags = filter(lambda x: x not in stop_words and len(x.strip()) > 0,
                  jieba.cut(st))
    tags = [tag for tag in tags if re.search(r"^\d+$", tag) is None]
    tag2gram = [' '.join(tags[i:i+2]) for i in range(len(tags)-1)]
    return tag2gram


def update_two_gram_tags():
    news_dict = search_data_match_all()
    news_list = news_dict['hits']['hits']
    for news in news_list:
        key = news['_source']['title'] + news['_source'].get(
            'description', [''])
        key = ' '.join(key)
        update = {"doc": {"two_gram_tags": sentence2tags(key)}}
        es.update(index='dongqiudi', doc_type='news', id=news['_id'],
                  body=update)
        # , version=1, version_type='internal'


def view_mapping():
    r = requests.get('http://localhost:9200/dongqiudi/_mapping/news')
    pprint.pprint(json.loads(r.content))


def word_cloud():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={
                            "aggs": {
                                "tagcloud": {
                                    "terms": {
                                        "field": "two_gram_tags.keyword",
                                        "size": 20
                                    }
                                }
                            },
                         })
    pprint.pprint(p['aggregations'])


def word_cloud2():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={
                        "query": {"bool": {"filter": {
                            "range": {"display_time": {
                                "gte": '2018-06-20',
                                "lte": '2018-06-20',
                                "format": "YYYY-MM-DD"}},
                            }}},

                        "aggs": {
                            "wordcloud": {
                                "terms": {
                                    "field": "two_gram_tags.keyword",
                                    "size": 20
                                }
                            }
                        },
                    })
    pprint.pprint(p['aggregations'])


def word_cloud3():
    p = es.search(index="dongqiudi",
                  doc_type="news",
                  body={
                        "query": {"bool": {"filter": {
                            "range": {"display_time": {
                                "gte": '2018-06-21 00:00:00',
                                "lte": '2018-06-22 00:00:00',
                                }},
                            }}},

                        "aggs": {
                            "wordcloud": {
                                "terms": {
                                    "field": "two_gram_tags.keyword",
                                    "size": 20
                                }
                            }
                        },
                    })
    pprint.pprint(p['aggregations'])


if __name__ == '__main__':

    # update_two_gram_tags()

    # count_data()

    # word_cloud()

    # word_cloud2()

    word_cloud3()
